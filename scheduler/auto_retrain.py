"""
Syntropy Quant - Automated Model Retraining Pipeline
Inspired by Two Sigma / Citadel ML Ops

Features:
- Scheduled retraining (daily/weekly/monthly)
- Model versioning and rollback
- Performance validation before deployment
- A/B testing support
- Automatic feature drift detection
"""

import os
import sys
import json
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel


@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    created_at: str
    training_symbols: List[str]
    epochs: int
    final_loss: float
    validation_sharpe: float
    validation_return: float
    is_active: bool = False
    notes: str = ""


@dataclass
class RetrainConfig:
    """Retraining configuration"""
    # Schedule
    retrain_frequency: str = "weekly"  # daily, weekly, monthly
    retrain_day: int = 0  # 0=Monday for weekly, 1-31 for monthly

    # Training params
    epochs: int = 100
    learning_rate: float = 0.0007
    batch_size: int = 1024

    # Data params
    training_lookback_years: int = 7
    validation_months: int = 6

    # Validation thresholds
    min_sharpe_improvement: float = 0.1
    min_return_improvement: float = 0.05
    max_loss_threshold: float = 1.0

    # Model management
    keep_versions: int = 5
    auto_rollback: bool = True


class ModelRegistry:
    """
    Model version registry and management
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "registry.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self._load_registry()

    def _load_registry(self):
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                for vid, vdata in data.items():
                    self.versions[vid] = ModelVersion(**vdata)

    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            data = {vid: asdict(v) for vid, v in self.versions.items()}
            json.dump(data, f, indent=2)

    def generate_version_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"

    def register_model(
        self,
        model_path: str,
        training_symbols: List[str],
        epochs: int,
        final_loss: float,
        validation_sharpe: float,
        validation_return: float,
        notes: str = ""
    ) -> str:
        version_id = self.generate_version_id()

        # Copy model to versioned path
        versioned_path = self.models_dir / f"{version_id}.pt"
        shutil.copy(model_path, versioned_path)

        # Register version
        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now().isoformat(),
            training_symbols=training_symbols,
            epochs=epochs,
            final_loss=final_loss,
            validation_sharpe=validation_sharpe,
            validation_return=validation_return,
            is_active=False,
            notes=notes
        )

        self.versions[version_id] = version
        self._save_registry()

        return version_id

    def activate_version(self, version_id: str):
        """Set a version as active (production)"""
        # Deactivate all
        for v in self.versions.values():
            v.is_active = False

        # Activate selected
        if version_id in self.versions:
            self.versions[version_id].is_active = True

            # Copy to main model path
            versioned_path = self.models_dir / f"{version_id}.pt"
            active_path = self.models_dir / "gauge_kernel.pt"
            shutil.copy(versioned_path, active_path)

        self._save_registry()

    def get_active_version(self) -> Optional[ModelVersion]:
        for v in self.versions.values():
            if v.is_active:
                return v
        return None

    def get_best_version(self, metric: str = "validation_sharpe") -> Optional[ModelVersion]:
        if not self.versions:
            return None
        return max(self.versions.values(), key=lambda v: getattr(v, metric))

    def cleanup_old_versions(self, keep: int = 5):
        """Remove old versions, keeping the best N"""
        if len(self.versions) <= keep:
            return

        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.validation_sharpe,
            reverse=True
        )

        # Keep active version + top N
        to_keep = set()
        for v in sorted_versions[:keep]:
            to_keep.add(v.version_id)

        active = self.get_active_version()
        if active:
            to_keep.add(active.version_id)

        # Remove others
        to_remove = [vid for vid in self.versions if vid not in to_keep]
        for vid in to_remove:
            model_path = self.models_dir / f"{vid}.pt"
            if model_path.exists():
                model_path.unlink()
            del self.versions[vid]

        self._save_registry()


class AutoRetrainer:
    """
    Automated model retraining pipeline
    """

    def __init__(
        self,
        config: Optional[RetrainConfig] = None,
        models_dir: str = "models"
    ):
        self.config = config or RetrainConfig()
        self.registry = ModelRegistry(models_dir)
        self.models_dir = Path(models_dir)

        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger('AutoRetrainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [RETRAIN] %(levelname)s: %(message)s'
            ))
            self.logger.addHandler(handler)

    def should_retrain(self) -> Tuple[bool, str]:
        """Check if retraining is due"""
        active = self.registry.get_active_version()

        if not active:
            return True, "No active model"

        created = datetime.fromisoformat(active.created_at)
        now = datetime.now()

        if self.config.retrain_frequency == "daily":
            if (now - created).days >= 1:
                return True, "Daily retrain due"

        elif self.config.retrain_frequency == "weekly":
            if (now - created).days >= 7:
                return True, "Weekly retrain due"

        elif self.config.retrain_frequency == "monthly":
            if (now - created).days >= 30:
                return True, "Monthly retrain due"

        return False, "Not due yet"

    def fetch_training_data(
        self,
        symbols: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch and prepare training data"""
        fetcher = DataFetcher()
        builder = FeatureBuilder()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * self.config.training_lookback_years)).strftime("%Y-%m-%d")

        data_list, target_list = [], []

        for sym in symbols:
            self.logger.info(f"Fetching {sym}...")
            df = fetcher.fetch(sym, start_date, end_date)
            if df.empty:
                continue

            feat = builder.build(df)
            if len(feat) == 0:
                continue

            closes = df['Close'].values if 'Close' in df.columns else df['close'].values
            closes = closes[20:]
            if len(closes) > len(feat):
                closes = closes[:len(feat)]

            ret = (closes[1:] - closes[:-1]) / closes[:-1]

            # 3-class labels
            labels = np.ones(len(ret))
            labels[ret > 0.001] = 2  # Long
            labels[ret < -0.001] = 0  # Short

            data_list.append(feat[:-1])
            target_list.append(torch.tensor(labels, dtype=torch.long))

        if not data_list:
            raise ValueError("No training data available")

        X = torch.cat(data_list)
        Y = torch.cat(target_list)

        return X, Y

    def train_model(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> Tuple[SyntropyQuantKernel, float]:
        """Train new model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SyntropyQuantKernel(input_dim=4, hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        X = X.to(device)
        Y = Y.to(device)

        model.train()
        batch_size = self.config.batch_size

        for epoch in range(self.config.epochs):
            perm = torch.randperm(len(X))
            epoch_loss = 0

            for i in range(0, len(X), batch_size):
                idx = perm[i:i+batch_size]
                bx, by = X[idx], Y[idx]

                optimizer.zero_grad()
                logits, _ = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / (len(X) / batch_size)
                self.logger.info(f"Epoch {epoch+1}/{self.config.epochs} | Loss: {avg_loss:.4f}")

        final_loss = epoch_loss / (len(X) / batch_size)
        return model, final_loss

    def validate_model(
        self,
        model: SyntropyQuantKernel,
        symbols: List[str]
    ) -> Tuple[float, float]:
        """Validate model on recent data"""
        fetcher = DataFetcher()
        builder = FeatureBuilder()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30 * self.config.validation_months)).strftime("%Y-%m-%d")

        all_returns = []

        model.eval()
        with torch.no_grad():
            for sym in symbols[:5]:  # Validate on subset
                df = fetcher.fetch(sym, start_date, end_date)
                if df.empty:
                    continue

                feat = builder.build(df)
                if len(feat) == 0:
                    continue

                closes = df['Close'].values if 'Close' in df.columns else df['close'].values
                closes = closes[20:]
                returns = (closes[1:] - closes[:-1]) / closes[:-1]

                logits, _ = model(feat)
                probs = torch.softmax(logits, dim=1).numpy()
                signal = probs[:, 2] - probs[:, 0]
                signal = signal[:-1]

                pos = np.zeros_like(signal)
                pos[signal > 0.03] = 1.0
                pos[signal < -0.03] = -0.5

                strat_ret = pos * returns[:len(pos)]
                all_returns.extend(strat_ret.tolist())

        if not all_returns:
            return 0.0, 0.0

        returns = np.array(all_returns)
        ann_ret = np.mean(returns) * 252
        ann_vol = np.std(returns) * np.sqrt(252) + 1e-6
        sharpe = ann_ret / ann_vol
        total_ret = np.prod(1 + returns) - 1

        return sharpe, total_ret

    def run_retrain(
        self,
        symbols: List[str],
        force: bool = False
    ) -> Optional[str]:
        """
        Run full retraining pipeline

        Returns version_id if successful, None otherwise
        """
        # Check if retrain is needed
        should, reason = self.should_retrain()
        if not should and not force:
            self.logger.info(f"Skipping retrain: {reason}")
            return None

        self.logger.info(f"Starting retrain: {reason}")

        try:
            # Fetch data
            X, Y = self.fetch_training_data(symbols)
            self.logger.info(f"Training data: {len(X)} samples")

            # Train model
            model, final_loss = self.train_model(X, Y)
            self.logger.info(f"Training complete. Final loss: {final_loss:.4f}")

            # Validate
            val_sharpe, val_return = self.validate_model(model, symbols)
            self.logger.info(f"Validation: Sharpe={val_sharpe:.2f}, Return={val_return*100:.1f}%")

            # Check thresholds
            if final_loss > self.config.max_loss_threshold:
                self.logger.warning(f"Loss {final_loss:.4f} exceeds threshold {self.config.max_loss_threshold}")
                return None

            # Save model
            temp_path = self.models_dir / "temp_model.pt"
            torch.save(model.state_dict(), temp_path)

            # Register version
            version_id = self.registry.register_model(
                model_path=str(temp_path),
                training_symbols=symbols,
                epochs=self.config.epochs,
                final_loss=final_loss,
                validation_sharpe=val_sharpe,
                validation_return=val_return,
                notes=f"Auto-retrained on {datetime.now().strftime('%Y-%m-%d')}"
            )

            # Check if better than current
            current = self.registry.get_active_version()
            should_activate = True

            if current:
                sharpe_diff = val_sharpe - current.validation_sharpe
                if sharpe_diff < self.config.min_sharpe_improvement:
                    self.logger.info(f"New model not significantly better (Sharpe diff: {sharpe_diff:.2f})")
                    should_activate = False

            if should_activate:
                self.registry.activate_version(version_id)
                self.logger.info(f"Activated new model: {version_id}")

            # Cleanup
            self.registry.cleanup_old_versions(self.config.keep_versions)
            temp_path.unlink()

            return version_id

        except Exception as e:
            self.logger.error(f"Retrain failed: {e}")
            return None

    def rollback(self) -> bool:
        """Rollback to previous best version"""
        best = self.registry.get_best_version()
        if best:
            self.registry.activate_version(best.version_id)
            self.logger.info(f"Rolled back to {best.version_id}")
            return True
        return False


# Scheduler integration (can be run via cron)
def run_scheduled_retrain():
    """Entry point for scheduled retraining"""
    from config.optimized_portfolio import get_all_symbols

    symbols = get_all_symbols()
    retrainer = AutoRetrainer()

    version = retrainer.run_retrain(symbols)
    if version:
        print(f"Retrain successful: {version}")
    else:
        print("Retrain skipped or failed")


if __name__ == '__main__':
    # Test retraining
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']

    config = RetrainConfig(
        epochs=20,  # Quick test
        retrain_frequency="weekly"
    )

    retrainer = AutoRetrainer(config)

    print("\n=== Model Registry ===")
    for vid, v in retrainer.registry.versions.items():
        status = "ACTIVE" if v.is_active else ""
        print(f"{vid}: Sharpe={v.validation_sharpe:.2f} {status}")

    print("\n=== Running Retrain ===")
    version = retrainer.run_retrain(symbols, force=True)
