"""
Two-Stage Inference Pipeline for Bike Defect Detection
Stage 1: Binary Classification (Intact vs Damaged)
Stage 2: DRAEM Defect Localization (if damaged)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional
import os
from dotenv import load_dotenv

from src.models.classifier import BikeClassifier
from inference_draem import DRAEMInference

# Load environment variables
load_dotenv()


class BikeDefectPipeline:
    """Two-stage bike defect detection pipeline"""

    def __init__(
        self,
        classifier_path: str,
        draem_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize the two-stage pipeline.

        Args:
            classifier_path: Path to Stage 1 classifier checkpoint
            draem_path: Path to Stage 2 DRAEM checkpoint (optional)
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        # -------------------------------
        # STAGE 1: Load Classifier
        # -------------------------------
        classifier_path = Path(classifier_path)
        if not classifier_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {classifier_path}")

        print("Loading Stage 1: Binary Classifier...")
        checkpoint = torch.load(
            classifier_path,
            map_location=self.device,
            weights_only=False
        )

        self.classifier = BikeClassifier(num_classes=2).to(self.device)

        if "model_state_dict" in checkpoint:
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.classifier.load_state_dict(checkpoint)

        self.classifier.eval()
        print(f"  Stage 1 loaded on {self.device}")

        # -------------------------------
        # STAGE 2: Load DRAEM (optional)
        # -------------------------------
        self.draem = None
        if draem_path:
            draem_path = Path(draem_path)
            if draem_path.exists():
                print("Loading Stage 2: DRAEM Localizer...")
                self.draem = DRAEMInference(
                    model_path=str(draem_path),
                    device=str(self.device),
                    image_size=256
                )
                print(f"  Stage 2 loaded on {self.device}")
            else:
                print(f"  Warning: DRAEM checkpoint not found: {draem_path}")
                print("  Stage 2 will be skipped")

        # Classifier transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("=" * 60)
        print("Pipeline Ready")
        print(f"  Stage 1: Binary Classification")
        print(f"  Stage 2: DRAEM Localization {'(enabled)' if self.draem else '(disabled)'}")
        print("=" * 60)

    def predict(
        self,
        image_path: str,
        visualize: bool = True,
        save_dir: str = "outputs",
        run_stage2: bool = True,
        draem_threshold: float = 0.6
    ) -> Dict:
        """
        Run the two-stage prediction pipeline.

        Args:
            image_path: Path to input image
            visualize: Whether to save visualization
            save_dir: Directory to save outputs
            run_stage2: Whether to run Stage 2 if damaged
            draem_threshold: Threshold for DRAEM heatmap refinement

        Returns:
            Dictionary with prediction results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # -------------------------------
        # STAGE 1: Binary Classification
        # -------------------------------
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()

        is_damaged = pred == 1
        confidence = probs[0, pred].item()

        result = {
            "stage1": {
                "is_damaged": is_damaged,
                "confidence": confidence,
                "probabilities": {
                    "intact": probs[0, 0].item(),
                    "damaged": probs[0, 1].item()
                }
            },
            "is_damaged": is_damaged,
            "confidence": confidence,
            "probabilities": {
                "intact": probs[0, 0].item(),
                "damaged": probs[0, 1].item()
            },
            "image_path": str(image_path),
            "stage2": None
        }

        print("=" * 60)
        print(f"IMAGE: {image_path.name}")
        print("-" * 60)
        print(f"STAGE 1 - Classification")
        print(f"  Prediction: {'DAMAGED' if is_damaged else 'INTACT'}")
        print(f"  Confidence: {confidence:.2%}")

        # -------------------------------
        # STAGE 2: DRAEM Localization
        # -------------------------------
        if is_damaged and run_stage2 and self.draem:
            print("-" * 60)
            print(f"STAGE 2 - Defect Localization")

            draem_result = self.draem.predict(
                image_np,
                use_bike_mask=True,
                refine_heatmap_flag=True,
                threshold=draem_threshold
            )

            result["stage2"] = {
                "anomaly_score": draem_result["anomaly_score"],
                "confidence": draem_result["confidence"],
                "heatmap": draem_result["heatmap"],
                "heatmap_overlay": draem_result["heatmap_overlay"],
                "reconstruction": draem_result["reconstruction"]
            }

            print(f"  Anomaly Score: {draem_result['anomaly_score']:.2%}")
            print(f"  Confidence: {draem_result['confidence']:.2%}")

        print("=" * 60)

        # Visualize and save
        output_path = None
        if visualize:
            output_path = self.visualize_and_save(
                image_np=image_np,
                result=result,
                image_name=image_path.stem,
                save_dir=save_dir
            )

        result["output_image"] = output_path
        return result

    def visualize_and_save(
        self,
        image_np: np.ndarray,
        result: Dict,
        image_name: str,
        save_dir: str
    ) -> str:
        """Visualize and save the prediction results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        is_damaged = result["is_damaged"]
        confidence = result["confidence"]
        stage2 = result.get("stage2")

        if is_damaged and stage2:
            # Two-panel visualization: Original + Heatmap overlay
            _, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Original image with classification result
            axes[0].imshow(image_np)
            axes[0].set_title(
                f"Stage 1: DAMAGED ({confidence:.2%})",
                fontsize=14,
                fontweight="bold",
                color="red"
            )
            axes[0].axis("off")

            # Heatmap overlay
            axes[1].imshow(stage2["heatmap_overlay"])
            axes[1].set_title(
                f"Stage 2: Defect Localization (Score: {stage2['anomaly_score']:.2%})",
                fontsize=14,
                fontweight="bold",
                color="red"
            )
            axes[1].axis("off")

            plt.tight_layout()

        else:
            # Single panel: Just classification result
            _, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image_np)

            status = "DAMAGED" if is_damaged else "INTACT"
            color = "red" if is_damaged else "green"
            ax.set_title(
                f"{status} ({confidence:.2%})",
                fontsize=14,
                fontweight="bold",
                color=color
            )
            ax.axis("off")

        output_path = save_dir / f"pred_{image_name}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"Output saved to: {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Two-stage bike defect detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--classifier",
        default=os.getenv("CLASSIFIER_PATH", "checkpoints/stage1/best_model.pth"),
        help="Stage 1 classifier checkpoint"
    )
    parser.add_argument(
        "--draem",
        default=os.getenv("DRAEM_PATH", "checkpoints/stage2/best_model.pth"),
        help="Stage 2 DRAEM checkpoint"
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--no-stage2", action="store_true", help="Skip Stage 2")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="DRAEM threshold")

    args = parser.parse_args()

    pipeline = BikeDefectPipeline(
        classifier_path=args.classifier,
        draem_path=args.draem if not args.no_stage2 else None,
        device=args.device
    )

    result = pipeline.predict(
        image_path=args.image,
        visualize=not args.no_vis,
        save_dir=args.outdir,
        run_stage2=not args.no_stage2,
        draem_threshold=args.threshold
    )

    # Print summary
    print("\nRESULT SUMMARY:")
    print(f"  Is Damaged: {result['is_damaged']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    if result["stage2"]:
        print(f"  Anomaly Score: {result['stage2']['anomaly_score']:.2%}")


if __name__ == "__main__":
    main()
