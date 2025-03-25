# %%
import os
import warnings
import time
import logging
import torch
from torch import nn, Tensor
import numpy as np
from torch.nn import functional as F
from typing import Tuple, Optional, Dict, Iterable
import gzip
import base64
from datetime import datetime
from contextlib import contextmanager
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.tensorboard.writer import SummaryWriter
import evaluate
from datasets import load_dataset
from transformers import WhisperTokenizer
import transformers
from tqdm.notebook import tqdm
from dataclasses import dataclass
from audio import pad
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

tokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small")

# %%


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= len(tokenizer)

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):

        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks


# %%

def ctx_to_samples(audio_ctx, hop_length):
    samples_token = hop_length * 2
    n_samples = audio_ctx * samples_token
    return n_samples

def load_wave(wave_data, sample_rate):
    if isinstance(wave_data, str):
        waveform, sr = torchaudio.load(uri=wave_data, normalize=False)
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sr = wave_data["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    if sr != sample_rate:
        original_length = waveform.shape[1]
        target_length = int(original_length * (sample_rate / sr))
        
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        
        if abs(waveform.shape[1] - target_length) > 1:
            new_waveform = torch.zeros((waveform.shape[0], target_length), dtype=waveform.dtype, device=waveform.device)
            copy_length = min(waveform.shape[1], target_length)
            new_waveform[:, :copy_length] = waveform[:, :copy_length]
            waveform = new_waveform
    
    return waveform.flatten()

def pad(array, target_length, axis=-1, dtype: torch.dtype = torch.float32):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(dtype)
    if torch.is_tensor(array):
        if array.shape[axis] > target_length:
            array = array.index_select(
                dim=axis,
                index=torch.arange(
                    end=target_length, device=array.device, dtype=torch.long
                ),
            )
        if array.shape[axis] < target_length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, target_length - array.shape[axis])
            array = F.pad(
                input=array, pad=[pad for sizes in pad_widths[::-1] for pad in sizes]
            )
        array = array.to(dtype=dtype)
    else:
        raise TypeError(
            f"Unsupported input type: {type(array)}. Expected torch.Tensor or np.ndarray."
        )
    return array

def exact_div(x, y):
    assert x % y == 0
    return x // y

def process_audio(audio, audio_ctx, mels, hop_length, n_fft, sr):

    audio = load_wave(wave_data=audio, sample_rate=sr)
    n_samples = ctx_to_samples(audio_ctx=audio_ctx, hop_length=hop_length)
    audio = pad(array=audio, target_length=n_samples)

    transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mels,
        norm='slaney',
        normalized=True,
        power=2.0,
        center=True, 
        window_fn=torch.hann_window,
    )
    
    mel_spectrogram = transform(audio)

    target_frames = exact_div(n_samples, hop_length) 
    mel_spectrogram = pad(array=mel_spectrogram, target_length=target_frames, axis=-1)

    log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    
    return log_mel

tokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small")

class DataCollator:
    def __init__(self, tokenizer, audio_ctx, text_ctx, mels, n_fft, hop_length, sample_rate=16000, device="cpu"):
        self.tokenizer = tokenizer
        self.text_ctx = text_ctx
        self.audio_ctx = audio_ctx
        self.sample_rate = sample_rate
        self.mels = mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.decoder_start_token_id = 50258
        self.pad_token_id = 50257

    def __call__(self, features):
        batch = len(features)
        
        max_time_frames = (
            ctx_to_samples(audio_ctx=self.audio_ctx, hop_length=self.hop_length)
            // self.hop_length
        )
        batch_audio = torch.zeros(
            size=(batch, self.mels, max_time_frames),
            dtype=torch.float32,
            device=self.device,
        )
        
        encoded_features = []
        
        for i, feature in enumerate(features):
            audio = process_audio(
                audio=feature["audio"],
                audio_ctx=self.audio_ctx,
                mels=self.mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                sr=self.sample_rate,
            )
            time_frames = audio.shape[-1]
            batch_audio[i, :, :time_frames] = torch.tensor(data=audio, dtype=torch.float32)
            
            transcript = feature["transcription"]
            encoded = self.tokenizer.encode(transcript)
            encoded_features.append({"input_ids": encoded})
        
        batch_encoded = self.tokenizer.pad(
            encoded_features, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.text_ctx
        )
        
        batch_input_ids = batch_encoded["input_ids"].to(self.device)
        
        batch_labels = batch_encoded["input_ids"].masked_fill(
            batch_encoded.attention_mask.ne(1), -100).to(self.device)
        
        if (batch_labels[:, 0] == self.decoder_start_token_id).all():
            batch_labels = batch_labels[:, 1:]
            batch_labels = torch.cat([
                batch_labels, 
                torch.full((batch, 1), -100, device=self.device)
            ], dim=1)
        
        return {
            "input_features": batch_audio,
            "input_ids": batch_input_ids,
            "labels": batch_labels,
        }


metric = evaluate.load(path="wer")
def compute_metrics(pred, tokenizer):
    pred_ids = pred["predictions"]
    label_ids = pred["label_ids"]
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    else:
        pred_ids = pred_ids
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def generate_predictions(model, input_features_encoded, tokenizer, device, batch_size, min_length):
    decoder_start_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    generated_ids = torch.full(
        size=(batch_size, 1), 
        fill_value=decoder_start_token_id, 
        dtype=torch.long, 
        device=device
    )
    
    max_length = 448
    all_sequences_finished = False
    
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for i in range(max_length - 1):
        attention_mask = torch.ones_like(generated_ids)
        
        with torch.no_grad():
            outputs = model(
                input_features=input_features_encoded,
                decoder_input_ids=generated_ids,
                decoder_attention_mask=attention_mask)
            
        next_token_logits = outputs.logits[:, -1, :]
        
        if i < min_length:
            next_token_logits[:, eos_token_id] = float('-inf')
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
        finished_sequences = finished_sequences | (next_tokens.squeeze(-1) == eos_token_id)
        if finished_sequences.all() and i >= min_length:
            break
    
    if generated_ids.size(1) < max_length:
        generated_ids = torch.nn.functional.pad(
            input=generated_ids, 
            pad=(0, max_length - generated_ids.size(1)), 
            value=pad_token_id)
    
    return generated_ids

def train_and_evaluate(model, tokenizer, train_loader, eval_loader, optimizer, scheduler, loss_fn, max_steps=10000, device='cuda', 
    accumulation_steps=1, clear_cache=True, log_interval=10, eval_interval=100, save_interval=1000, checkpoint_dir="checkpoint_dir", log_dir="log_dir"):
    
    model.to(device)
    global_step = 0
    scaler = torch.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    train_iterator = iter(train_loader)
    total_loss = 0
    step_in_report = 0
    dataset_epochs = 0

    progress_bar = tqdm(total=max_steps, desc="Training Progress", leave=True, colour='green')

    model.train()
    optimizer.zero_grad()

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True, profile_memory=True, with_stack=True)

    profiler.start()

    while global_step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            dataset_epochs += 1
            print(f"Starting dataset epoch {dataset_epochs}")

            if step_in_report > 0:
                avg_loss = total_loss / step_in_report
                logging.info(f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0
                step_in_report = 0

        start_time = time.time()

        input_features = batch['input_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].long().to(device)

        with torch.autocast(device_type="cuda"):
            input_features_encoded = model.encoder(input_features)
            decoder_output = model.decoder(input_ids, input_features_encoded)
            logits = decoder_output.view(-1, decoder_output.size(-1))
            active_logits = logits.view(-1, decoder_output.size(-1))
            active_labels = labels.view(-1)
            active_mask = active_labels != -100
            active_logits = active_logits[active_mask]
            active_labels = active_labels[active_mask]
            loss = loss_fn(active_logits, active_labels)

        total_loss += loss.item()
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if clear_cache:
                torch.cuda.empty_cache()

        end_time = time.time()
        samples_per_sec = len(batch['input_features']) / (end_time - start_time)

        if global_step % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar(tag='Loss/train', scalar_value=total_loss / (global_step + 1), global_step=global_step)
            writer.add_scalar(tag='LearningRate', scalar_value=lr, global_step=global_step)
            writer.add_scalar(tag='SamplesPerSec', scalar_value=samples_per_sec, global_step=global_step)
            logging.info(f"Step {global_step} - Loss: {total_loss / (global_step + 1):.4f}, LR: {lr:.8f}")
        
        if global_step % eval_interval == 0:
            model.eval()
            eval_start_time = time.time()
            eval_loss = 0
            all_predictions = []
            all_labels = []
            batch_count = 0
            total_samples = 0

            with torch.no_grad():
                for eval_batch in eval_loader:
                    input_features = eval_batch['input_features'].to(device)
                    input_ids = eval_batch['input_ids'].to(device)
                    labels = eval_batch['labels'].long().to(device)

                    batch_size = input_features.size(0)
                    total_samples += batch_size

                    input_features_encoded = model.encoder(input_features)
                    decoder_output = model.decoder(input_ids, input_features_encoded)
                    logits = decoder_output.view(-1, decoder_output.size(-1))
                    loss = loss_fn(logits, labels.view(-1))
                    eval_loss += loss.item()

                    all_predictions.extend(torch.argmax(decoder_output, dim=-1).cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    batch_count += 1

            lr = scheduler.get_last_lr()[0]
            eval_time = time.time() - eval_start_time
            loss_avg = eval_loss / batch_count if batch_count > 0 else 0
            predictions = {"predictions": np.array(all_predictions, dtype=object), "label_ids": np.array(all_labels, dtype=object)}
            metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

            writer.add_scalar('Loss/eval', loss_avg, global_step)
            writer.add_scalar('WER', metrics['wer'], global_step)
            writer.add_scalar('EvalSamples', total_samples, global_step)
            writer.add_scalar('EvalTimeSeconds', eval_time, global_step)

            pred = tokenizer.decode(all_predictions[0], skip_special_tokens=True)
            label = tokenizer.decode(all_labels[0], skip_special_tokens=True)
            
            print(f"STEP {global_step} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}")
            print(f"PRED: '{pred}'")
            print(f"REF : '{label}'")
            print()
            logging.info(f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}")
            model.train()

        if global_step % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")
            
        global_step += 1
        step_in_report += 1

        avg_loss = total_loss / (global_step + 1)
        postfix_dict = {'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.6f}', 'WER': f'{metrics["wer"]:.4f}' if 'wer' in metrics else 'N/A',
            'samp/sec': f'{samples_per_sec:.1f}'}
        progress_bar.set_postfix(postfix_dict, refresh=True)
        progress_bar.update(1)
        scheduler.step()
        profiler.step()
        
    profiler.stop()
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed after {global_step} steps. Final model saved to {final_model_path}")
    writer.close()
    progress_bar.close()

checkpoint_dir = './output/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = os.path.join("./output/logs", datetime.now().strftime(format="%m-%d_%H"))
os.makedirs(name=log_dir, exist_ok=True)

# %%

if __name__ == "__main__":

    checkpoint_dir = './output/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join("./output/logs", datetime.now().strftime(format="%m-%d_%H"))
    os.makedirs(name=log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'), filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    token = "" #hugging face token
    dataset = load_dataset(
        path="google/fleurs",
        name="en_us",
        streaming=False,
        token=token,
        trust_remote_code=True,
        cache_dir="E:/cache",
    ).select_columns(column_names=["audio", "transcription"])
   
    debug = None
    
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=512,
        n_audio_head=4,
        n_audio_layer=4,
        n_vocab=len(tokenizer),
        n_text_ctx=448,
        n_text_state=512,
        n_text_head=4,
        n_text_layer=4,
    )
    
    model = Whisper(dims).to('cuda')

    DataCollator = DataCollator(tokenizer=tokenizer, 
                                audio_ctx=dims.n_audio_ctx, 
                                text_ctx=dims.n_text_ctx, 
                                mels=dims.n_mels,
                                n_fft=400,
                                hop_length=160,
                                sample_rate=16000, 
                                device='cuda')

    train_dataloader = DataLoader(
        dataset=dataset["train"], 
        batch_size=1, 
        collate_fn=DataCollator,
        num_workers=0)

    eval_dataloader = DataLoader(
        dataset=dataset["test"],
        batch_size=1,
        collate_fn=DataCollator,
        num_workers=0)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, total_iters=10000, last_epoch=-1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
    train_and_evaluate(model=model, 
        tokenizer=tokenizer, 
        train_loader=train_dataloader, 
        eval_loader=eval_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn, 
        max_steps=10000,
        device='cuda', 
        accumulation_steps=1, 
        clear_cache=False, 
        log_interval=10, 
        eval_interval=500, 
        save_interval=10000, 
        checkpoint_dir=checkpoint_dir, 
        log_dir=log_dir
        )
    

# %%
from tensorboard import program
log_dir = "./output/logs" 
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"TensorBoard started at {url}")


