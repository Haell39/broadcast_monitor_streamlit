import time
import threading
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import cv2
import streamlit as st
from skimage.metrics import structural_similarity as ssim

try:
    import av  # PyAV para análise de áudio
except Exception:
    av = None

# ==========================
# Utilitários e Estruturas
# ==========================
@dataclass
class Event:
    t_wall: float  # epoch seconds (parede)
    kind: str      # e.g., "video_black", "video_freeze", "audio_silence"
    severity: str  # "info" | "warn" | "crit"
    message: str   # texto humano


def now_ts() -> float:
    return time.time()


def fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# ==========================
# Detectores de Vídeo
# ==========================
class VideoDetectors:
    def __init__(self,
                 black_luma_thresh: float = 20.0,
                 white_luma_thresh: float = 235.0,
                 min_black_seconds: float = 2.0,
                 min_white_seconds: float = 1.5,
                 freeze_ssim_thresh: float = 0.995,
                 min_freeze_seconds: float = 2.0,
                 fps_hint: Optional[float] = None):
        self.black_luma_thresh = black_luma_thresh
        self.white_luma_thresh = white_luma_thresh
        self.min_black_seconds = min_black_seconds
        self.min_white_seconds = min_white_seconds
        self.freeze_ssim_thresh = freeze_ssim_thresh
        self.min_freeze_seconds = min_freeze_seconds
        self.prev_gray = None
        self.last_not_frozen_ts = None
        self.black_start_ts = None
        self.white_start_ts = None
        self.fps_hint = fps_hint

    def analyze(self, frame: np.ndarray, t: float) -> List[Event]:
        events = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        std_luma = float(np.std(gray))

        # Black frame logic
        if mean_luma < self.black_luma_thresh and std_luma < 10.0:
            if self.black_start_ts is None:
                self.black_start_ts = t
            duration = t - self.black_start_ts
            if duration >= self.min_black_seconds:
                events.append(Event(t, "video_black", "crit",
                                    f"Vídeo escuro/preto há {duration:.1f}s (luma={mean_luma:.1f}, std={std_luma:.1f})."))
        else:
            self.black_start_ts = None

        # White-out simples (brilho muito alto e baixa variação)
        if mean_luma > self.white_luma_thresh and std_luma < 10.0:
            if self.white_start_ts is None:
                self.white_start_ts = t
            duration = t - self.white_start_ts
            if duration >= self.min_white_seconds:
                events.append(Event(t, "video_whiteout", "warn",
                                    f"Quadro claro/estourado há {duration:.1f}s (luma={mean_luma:.1f})."))
        else:
            self.white_start_ts = None

        # Freeze via SSIM com frame anterior
        if self.prev_gray is not None:
            try:
                s = ssim(self.prev_gray, gray)
            except Exception:
                s = 0.0
            if s >= self.freeze_ssim_thresh:
                # congelado
                if self.last_not_frozen_ts is None:
                    self.last_not_frozen_ts = t
                frozen_for = t - self.last_not_frozen_ts
                if frozen_for >= self.min_freeze_seconds:
                    events.append(Event(t, "video_freeze", "crit",
                                        f"Quadro congelado há {frozen_for:.1f}s (SSIM={s:.4f})."))
            else:
                # movimento detectado
                self.last_not_frozen_ts = t
        else:
            self.last_not_frozen_ts = t

        self.prev_gray = gray
        return events


# ==========================
# Detectores de Áudio
# ==========================
class AudioDetectors:
    def __init__(self,
                 silence_rms_thresh: float = 0.005,
                 min_silence_seconds: float = 2.5,
                 clip_ratio_thresh: float = 0.02,
                 min_clip_seconds: float = 1.0,
                 balance_ratio_warn: float = 6.0):
        """
        silence_rms_thresh: RMS normalizado (0..1) abaixo do qual consideramos silêncio
        clip_ratio_thresh: fração de amostras com |x| > 0.98 considerada "clipando"
        balance_ratio_warn: diferença em dB entre L e R que dispara alerta
        """
        self.silence_rms_thresh = silence_rms_thresh
        self.min_silence_seconds = min_silence_seconds
        self.clip_ratio_thresh = clip_ratio_thresh
        self.min_clip_seconds = min_clip_seconds
        self.balance_ratio_warn = balance_ratio_warn
        self.silence_start_ts = None
        self.clip_start_ts = None

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    def analyze_chunk(self, samples: np.ndarray, t: float, sr: int) -> List[Event]:
        """
        samples: float32 em [-1,1], shape (n,) mono ou (n, 2) estéreo
        """
        events = []
        if samples.ndim == 1:
            mono = samples
            l = r = mono
        else:
            l = samples[:, 0]
            r = samples[:, 1] if samples.shape[1] > 1 else samples[:, 0]
            mono = 0.5 * (l + r)

        rms_val = self._rms(mono)
        # Silêncio prolongado
        if rms_val < self.silence_rms_thresh:
            if self.silence_start_ts is None:
                self.silence_start_ts = t
            dur = t - self.silence_start_ts
            if dur >= self.min_silence_seconds:
                events.append(Event(t, "audio_silence", "crit",
                                    f"Silêncio no áudio há {dur:.1f}s (RMS={rms_val:.4f})."))
        else:
            self.silence_start_ts = None

        # Clipping prolongado
        clip_ratio = float(np.mean(np.abs(mono) > 0.98)) if mono.size else 0.0
        if clip_ratio > self.clip_ratio_thresh:
            if self.clip_start_ts is None:
                self.clip_start_ts = t
            dur = t - self.clip_start_ts
            if dur >= self.min_clip_seconds:
                events.append(Event(t, "audio_clipping", "warn",
                                    f"Possível clipping há {dur:.1f}s (amostras>0.98: {clip_ratio*100:.1f}%)."))
        else:
            self.clip_start_ts = None

        # Desbalanceamento L/R (diferença RMS em dB)
        rms_l = self._rms(l)
        rms_r = self._rms(r)
        if rms_l > 0 and rms_r > 0:
            db_diff = 20.0 * np.log10(max(rms_l, rms_r) / max(1e-12, min(rms_l, rms_r)))
            if db_diff >= self.balance_ratio_warn:
                side = "L" if rms_l > rms_r else "R"
                events.append(Event(t, "audio_balance", "warn",
                                    f"Desbalanceamento de canais (△{db_diff:.1f} dB, lado {side} mais alto)."))
        return events


# ==========================
# Pipelines de captura
# ==========================
class VideoMonitor:
    def __init__(self, source: str, enable_audio: bool = True,
                 vconf: Optional[dict] = None, aconf: Optional[dict] = None):
        self.source = source
        self.enable_audio = enable_audio and (av is not None)
        self.vdet = VideoDetectors(**(vconf or {}))
        self.adet = AudioDetectors(**(aconf or {}))
        self.events: List[Event] = []
        self._stopping = threading.Event()
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self.fps: Optional[float] = None

    def start(self):
        self._stopping.clear()
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()
        if self.enable_audio:
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()

    def stop(self):
        self._stopping.set()
        if self._video_thread:
            self._video_thread.join(timeout=5.0)
        if self._audio_thread:
            self._audio_thread.join(timeout=5.0)

    # ------ VIDEO ------
    def _video_loop(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.events.append(Event(now_ts(), "error", "crit",
                                     f"Falha ao abrir fonte de vídeo: {self.source}"))
            return
        # Tenta ler FPS da fonte
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0 and not np.isnan(fps):
            self.fps = float(fps)
            self.vdet.fps_hint = self.fps
        frame_interval = 1.0 / self.fps if self.fps else 0.04

        while not self._stopping.is_set():
            ret, frame = cap.read()
            t = now_ts()
            if not ret:
                self.events.append(Event(t, "video_read_error", "crit",
                                         "Leitura de quadro falhou (fim de arquivo ou perda de sinal)."))
                break
            # Reduz processamento (opcional): redimensiona
            h, w = frame.shape[:2]
            if max(h, w) > 720:
                scale = 720.0 / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

            # Analisar e armazenar events
            for ev in self.vdet.analyze(frame, t):
                self.events.append(ev)

            # Renderização no Streamlit: colocamos frame atual no placeholder
            st.session_state["last_frame"] = frame

            # Dorme para respeitar taxa
            time.sleep(frame_interval)

        cap.release()

    # ------ ÁUDIO ------
    def _audio_loop(self):
        if av is None:
            self.events.append(Event(now_ts(), "audio_disabled", "info",
                                     "PyAV não disponível; detecções de áudio desativadas."))
            return
        try:
            container = av.open(self.source)
        except Exception as e:
            self.events.append(Event(now_ts(), "audio_open_error", "warn",
                                     f"Não foi possível abrir áudio: {e}"))
            return
        stream = None
        for s in container.streams:
            if s.type == "audio":
                stream = s
                break
        if stream is None:
            self.events.append(Event(now_ts(), "audio_stream_missing", "warn",
                                     "Sem trilha de áudio detectada."))
            return

        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=48000)
        window_samples = 48000  # 1 segundo
        buf = []
        try:
            for packet in container.demux(stream):
                if self._stopping.is_set():
                    break
                for frame in packet.decode():
                    af = resampler.resample(frame)
                    # Converte para numpy float32 [-1,1]
                    arr = af.to_ndarray()
                    # arr shape: (channels, samples)
                    arr = arr.T  # (samples, channels)
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    buf.append(arr)
                    total = np.concatenate(buf, axis=0)
                    while total.shape[0] >= window_samples:
                        chunk = total[:window_samples]
                        total = total[window_samples:]
                        t = now_ts()
                        for ev in self.adet.analyze_chunk(chunk, t, 48000):
                            self.events.append(ev)
                    buf = [total]
        except Exception as e:
            self.events.append(Event(now_ts(), "audio_decode_error", "warn",
                                     f"Erro na decodificação de áudio: {e}"))


# ==========================
# Streamlit UI Helpers
# ==========================
_frame_placeholder = None
_chat_container = None


def _init_streamlit_ui():
    st.set_page_config(page_title="Monitor de Sinal – Protótipo", layout="wide")
    st.title("📺 Monitor de Integridade de Sinal – Protótipo IA")
    st.caption("Analisa vídeo/áudio em tempo quase real e envia alertas como chat.")

    with st.sidebar:
        st.subheader("Fonte de Sinal")
        source_mode = st.radio("Escolha a fonte:", ["Arquivo local", "URL/Stream"], index=0)
        file_source = None
        url_source = None
        if source_mode == "Arquivo local":
            up = st.file_uploader("Envie um arquivo de vídeo", type=["mp4", "mkv", "mov", "avi", "ts", "m3u8"])
            if up is not None:
                # Salva arquivo temporário
                import tempfile, os
                tmpdir = tempfile.gettempdir()
                fpath = os.path.join(tmpdir, up.name)
                with open(fpath, "wb") as f:
                    f.write(up.read())
                file_source = fpath
        else:
            url_source = st.text_input("URL/RTSP/HLS", placeholder="rtsp://... ou https://...m3u8")

        st.subheader("Parâmetros de Detecção")
        black_luma = st.slider("Limiar luma preto", 0, 60, 20)
        min_black = st.slider("Mín. segundos de preto", 1.0, 10.0, 2.0)
        white_luma = st.slider("Limiar luma branco", 200, 255, 235)
        min_white = st.slider("Mín. segundos de branco", 1.0, 10.0, 1.5)
        freeze_ssim = st.slider("SSIM p/ freeze (maior = mais sensível)", 0.90, 0.999, 0.995)
        min_freeze = st.slider("Mín. segundos de freeze", 0.5, 10.0, 2.0)

        st.markdown("**Áudio**")
        enable_audio = st.checkbox("Ativar detecções de áudio (requer PyAV)", value=True)
        silence_rms = st.slider("RMS p/ silêncio", 0.0, 0.05, 0.005)
        min_silence = st.slider("Mín. segundos de silêncio", 0.5, 10.0, 2.5)
        clip_ratio = st.slider("% amostras clipadas p/ alerta", 0.0, 0.10, 0.02)
        min_clip = st.slider("Mín. segundos clipando", 0.0, 5.0, 1.0)
        bal_db = st.slider("Desbalanceamento L/R (dB)", 1.0, 12.0, 6.0)

        start = st.button("▶ Iniciar Monitoramento")
        stop = st.button("⏹ Parar")

    global _frame_placeholder, _chat_container
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Preview do Vídeo")
        _frame_placeholder = st.empty()
    with cols[1]:
        st.subheader("Alertas (Chat)")
        _chat_container = st.container()

    return {
        "start": start,
        "stop": stop,
        "source": file_source or url_source,
        "vconf": {
            "black_luma_thresh": float(black_luma),
            "white_luma_thresh": float(white_luma),
            "min_black_seconds": float(min_black),
            "min_white_seconds": float(min_white),
            "freeze_ssim_thresh": float(freeze_ssim),
            "min_freeze_seconds": float(min_freeze),
        },
        "aconf": {
            "silence_rms_thresh": float(silence_rms),
            "min_silence_seconds": float(min_silence),
            "clip_ratio_thresh": float(clip_ratio),
            "min_clip_seconds": float(min_clip),
            "balance_ratio_warn": float(bal_db),
        },
        "enable_audio": enable_audio,
    }


def _drain_events_to_chat(monitor: VideoMonitor):
    if _chat_container is None:
        return
    last_idx = st.session_state.get("last_event_idx", 0)
    for ev in monitor.events[last_idx:]:
        with _chat_container:
            role = "assistant" if ev.severity in ("warn", "crit") else "user"
            icon = "⚠️" if ev.severity == "warn" else ("🛑" if ev.severity == "crit" else "ℹ️")
            st.chat_message(role).write(f"{icon} **{ev.kind}** @ {fmt_ts(ev.t_wall)} – {ev.message}")
    st.session_state.last_event_idx = len(monitor.events)


# ==========================
# Main App
# ==========================
if "monitor" not in st.session_state:
    st.session_state.monitor = None
if "running" not in st.session_state:
    st.session_state.running = False
if "last_event_idx" not in st.session_state:
    st.session_state.last_event_idx = 0

ui = _init_streamlit_ui()

if ui["start"]:
    if ui["source"]:
        if st.session_state.monitor is not None:
            st.session_state.monitor.stop()
        st.session_state.monitor = VideoMonitor(
            source=ui["source"],
            enable_audio=ui["enable_audio"],
            vconf=ui["vconf"],
            aconf=ui["aconf"],
        )
        st.session_state.monitor.start()
        st.session_state.running = True
        st.session_state.last_event_idx = 0
        st.success("Monitoramento iniciado.")
    else:
        st.error("Forneça um arquivo ou URL de vídeo.")

if ui["stop"]:
    if st.session_state.monitor is not None:
        st.session_state.monitor.stop()
        st.session_state.monitor = None
    st.session_state.running = False
    st.info("Monitoramento interrompido.")

if st.session_state.running and st.session_state.monitor is not None:
    _drain_events_to_chat(st.session_state.monitor)

    # Renderiza último frame salvo pela thread de captura
    if "last_frame" in st.session_state and st.session_state["last_frame"] is not None:
        rgb = cv2.cvtColor(st.session_state["last_frame"], cv2.COLOR_BGR2RGB)
        if _frame_placeholder is not None:
            _frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

    # Notinha de status
    if st.session_state.monitor.fps:
        st.caption(f"FPS estimado: {st.session_state.monitor.fps:.2f}")
    else:
        st.caption("FPS estimado: N/D")

    # Atualiza a UI em tempo real
    time.sleep(0.5)  # Taxa de atualização (ajuste se necessário)
    st.rerun()

# Exportar logs (JSONL)
evs = st.session_state.monitor.events if st.session_state.monitor else []
import json
log_bytes = ("\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in evs)).encode("utf-8")
st.download_button("📥 Baixar log (JSONL)", data=log_bytes, file_name="monitor_log.jsonl", mime="application/json")