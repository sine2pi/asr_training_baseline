import os
import time
import logging
from typing import Tuple, Optional, Dict, Union, Any, List
import gzip
import base64
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from dataclasses import dataclass
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.tensorboard.writer import SummaryWriter
import torchaudio.transforms as T
from torch.nn import functional as F
import torch, torchaudio
from torch import nn, Tensorimport torch, torchaudio
from torch import nn, Tensor
from contextlib import contextmanager
from typing import Iterable
from whisper.decoding import decode as decode_function
from whisper.decoding import detect_language as detect_language_function
from whisper.transcribe import transcribe as transcribe_function

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


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
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx), the mel spectrogram of the audio
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
        """        x : torch.LongTensor, shape = (batch_size, <= n_ctx), xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)"""
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

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "Embedding": 0}

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            if isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            if isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                self.init_counts["LayerNorm"] += 1
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
                self.init_counts["Embedding"] += 1
            
    def init_weights(self):  # noqa: F811
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            print(f"{module_type}: {count}")

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, input_features: torch.Tensor):
        return self.encoder(input_features)

    def logits(self,input_ids: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(input_ids, audio_features)


    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):

        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
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

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

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
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(
            waveform
        )
    return waveform.flatten()

def pad(array, target_length, axis=-1, dtype: torch.dtype = torch.float32):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(ndarray=array).to(dtype=dtype)
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

def process_audio(audio, audio_ctx, n_mels, hop_length, n_fft, sr):
    audio = load_wave(wave_data=audio, sample_rate=sr)
    target_length = ctx_to_samples(audio_ctx=audio_ctx, hop_length=hop_length)
    audio = pad(array=audio, target_length=target_length)
    transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        normalized=False,
        center=True,
    )
    mel_spectrogram = transform(audio)
    mel_spectrogram = mel_spectrogram[:, :3000]
    epsilon = 1e-10
    log_mel = torch.log(mel_spectrogram + epsilon)
    log_mel = (log_mel - log_mel.mean(dim=-1, keepdim=True)) / (log_mel.std(dim=-1, keepdim=True) + 1e-10)
    return log_mel

tokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small"
)

class DataCollator:
    def __init__(self, tokenizer, audio_ctx=1500, text_ctx=448, n_mels=128, n_fft=400, hop_length=160, sample_rate=16000, device="cpu"):
        self.tokenizer = tokenizer
        self.text_ctx = text_ctx
        self.audio_ctx = audio_ctx
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.decoder_start_token_id = tokenizer.bos_token_id + 1
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        batch = len(features)

        max_time_frames = (
            ctx_to_samples(audio_ctx=self.audio_ctx, hop_length=self.hop_length)
            // self.hop_length
        )
        batch_audio = torch.zeros(
            size=(batch, self.n_mels, max_time_frames),
            dtype=torch.float32,
            device=self.device,
        )
        batch_input_ids = torch.full(
            size=(batch, self.text_ctx),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        batch_labels = torch.full(
            size=(batch, self.text_ctx),
            fill_value=-100,
            dtype=torch.long,
            device=self.device,
        )

        for i, feature in enumerate(iterable=features):

            audio = process_audio(
                audio=feature["audio"],
                audio_ctx=self.audio_ctx,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                sr=self.sample_rate,
            )

            time_frames = audio.shape[-1]

            transcript = feature["transcription"]
            encoded_input = self.tokenizer.encode(transcript)
            encoded_label = self.tokenizer.encode(transcript)

            decoder_input = [self.decoder_start_token_id] + encoded_input
            labels = encoded_label + [self.tokenizer.eos_token_id]

            decoder_input = decoder_input[: self.text_ctx] + [self.pad_token_id] * (
                self.text_ctx - len(decoder_input)
            )
            labels = labels[: self.text_ctx] + [-100] * (self.text_ctx - len(labels))

            batch_input_ids[i, : len(decoder_input)] = torch.tensor(data=decoder_input, dtype=torch.long)
            batch_labels[i, : len(labels)] = torch.tensor(data=labels, dtype=torch.long)
            batch_audio[i, :, :time_frames] = torch.tensor(data=audio, dtype=torch.float32)

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
    wer = 100 * metric.compute(predictions=pred_str, references=label_str) # type: ignore
    return {"wer": wer}


def train_and_evaluate(model, tokenizer, train_loader, eval_loader, optimizer, scheduler, loss_fn,
                        max_steps=10000, device='cuda', accumulation_steps=1, clear_cache=True,
                        log_interval=10, eval_interval=100, save_interval=1000,
                        checkpoint_dir="checkpoint_dir", log_dir="log_dir"):
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
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=1,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

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
            writer.add_scalar(tag='Loss/train', scalar_value=total_loss / (global_step + 1), global_step=global_step)
            lr = optimizer.param_groups[0].get('lr', None)
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

  
            eval_time = time.time() - eval_start_time
            loss_avg = eval_loss / batch_count if batch_count > 0 else 0
            predictions = {"predictions": np.array(all_predictions, dtype=object), "label_ids": np.array(all_labels, dtype=object)}
            metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

            writer.add_scalar('Loss/eval', loss_avg, global_step)
            writer.add_scalar('WER', metrics['wer'], global_step)
            writer.add_scalar('EvalSamples', total_samples, global_step)
            writer.add_scalar('EvalTimeSeconds', eval_time, global_step)
            lr = scheduler.get_last_lr()[0]

            print(f" • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}")
            logging.info(f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}")
            #scheduler.step(metrics['wer'])
            model.train()

        if global_step % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")
            
        global_step += 1
        step_in_report += 1

        lr = scheduler.get_last_lr()[0]
        avg_loss = total_loss / (global_step + 1)
        postfix_dict = {
            'loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.6f}',
            'WER': f'{metrics["wer"]:.4f}' if 'wer' in metrics else 'N/A',
            'samp/sec': f'{samples_per_sec:.1f}'
        }
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

if __name__ == "__main__":

    checkpoint_dir = './output/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join("./output/logs", datetime.now().strftime(format="%m-%d_%H"))
    os.makedirs(name=log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'), filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    token = ""
    dataset = load_dataset(
        path="google/fleurs",
        name="en_us",
        streaming=False,
        token=token,
        trust_remote_code=True,
        cache_dir="E:/cache",
    ).select_columns(column_names=["audio", "transcription"])

    DataCollator = DataCollator(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset=dataset["train"], 
        batch_size=1, 
        collate_fn=DataCollator,
        num_workers=0)

    eval_dataloader = DataLoader(
        dataset=dataset["test"].take(100),
        batch_size=1,
        collate_fn=DataCollator,
        num_workers=0)
    
    dims =ModelDimensions(
        n_mels=128,
        n_audio_ctx=1500,
        n_audio_head=2,
        n_audio_layer=2,
        n_audio_state=512,
        n_vocab=51865,
        n_text_state=512,
        n_text_ctx=448,
        n_text_head=2,
        n_text_layer=2,
    )
    model = Whisper(dims=dims).to(device='cuda')
    model.init_weights()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=100000, last_epoch=-1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_and_evaluate(model=model, 
        tokenizer=tokenizer, 
        train_loader=train_dataloader, 
        eval_loader=eval_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn, 
        max_steps=100000,
        device='cuda', 
        accumulation_steps=1, 
        clear_cache=True, 
        log_interval=10, 
        eval_interval=100, 
        save_interval=25000, 
        checkpoint_dir=checkpoint_dir, 
        log_dir=log_dir
        )

