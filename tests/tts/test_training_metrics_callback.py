import math

from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.integrations.integration_utils import rewrite_logs

from bodhan_genai.tts.training.callbacks import TrainingMetricsCallback


def test_training_metrics_callback_uses_unprefixed_train_keys(tmp_path):
    callback = TrainingMetricsCallback(
        num_params=1_000_000,
        max_seq_len=1024,
        gradient_accumulation_steps=2,
        peak_tflops_per_gpu=100.0,
    )
    args = TrainingArguments(output_dir=str(tmp_path))
    control = TrainerControl()

    first_logs = {"loss": 2.0}
    callback.on_log(
        args=args,
        state=TrainerState(global_step=10),
        control=control,
        logs=first_logs,
    )

    assert "perplexity" in first_logs
    assert "train/perplexity" not in first_logs
    assert math.isclose(first_logs["perplexity"], math.exp(2.0))
    assert rewrite_logs(first_logs)["train/perplexity"] == first_logs["perplexity"]

    second_logs = {"loss": 2.0}
    callback._last_log_time = 100.0
    callback._last_step = 10
    callback.on_log(
        args=args,
        state=TrainerState(global_step=20),
        control=control,
        logs=second_logs,
    )

    assert "mfu" in second_logs
    assert "tflops_achieved" in second_logs
    assert "tflops_aggregate" in second_logs
    assert "train/mfu" not in second_logs
    rewritten = rewrite_logs(second_logs)
    assert "train/mfu" in rewritten
    assert "train/tflops_achieved" in rewritten
    assert "train/tflops_aggregate" in rewritten


def test_training_metrics_callback_uses_eval_prefix_style_for_eval_metrics(tmp_path):
    callback = TrainingMetricsCallback(
        num_params=1_000_000,
        max_seq_len=1024,
        peak_tflops_per_gpu=100.0,
    )
    args = TrainingArguments(output_dir=str(tmp_path))
    control = TrainerControl()
    logs = {"eval_loss": 3.0}

    callback.on_log(
        args=args,
        state=TrainerState(global_step=10),
        control=control,
        logs=logs,
    )

    assert "eval_perplexity" in logs
    assert "eval/perplexity" not in logs
    assert math.isclose(logs["eval_perplexity"], math.exp(3.0))
    assert rewrite_logs(logs)["eval/perplexity"] == logs["eval_perplexity"]
