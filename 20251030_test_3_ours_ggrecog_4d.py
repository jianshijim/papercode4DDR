import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import warnings
from datetime import datetime
import sys

from train_3_ours import (
    DrivingBehaviorDataset,
    OptimizedStudentModel,
    LightweightSensorEncoder,
    EfficientVideoEncoder,
EfficientCrossModalAttention,
collate_fn
)

warnings.filterwarnings('ignore')

# seed
torch.manual_seed(42)
np.random.seed(42)


class OptimizedStudentModelWithModalFlags(OptimizedStudentModel):
    
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes)

        self.sensor_null_token = nn.Parameter(torch.randn(1, 64))
        self.integral_null_token = nn.Parameter(torch.randn(1, 64))
        self.video_null_token = nn.Parameter(torch.randn(1, 128))
        

        nn.init.xavier_uniform_(self.sensor_null_token)
        nn.init.xavier_uniform_(self.integral_null_token)
        nn.init.xavier_uniform_(self.video_null_token)
    
    def forward(self, sensor_data, integral_data, video_data, video_lengths=None,
                use_sensor=True, use_integral=True, use_video=True, return_features=False):
        """
        Args:
            sensor_data
            integral_data
            video_data
            video_lengths
            use_sensor
            use_integral:
            use_video
            return_features
        """

        if use_sensor and sensor_data is not None:
            sensor_mask = (sensor_data.abs().sum(dim=-1) == 0)
            sensor_feat = self.sensor_encoder(sensor_data, sensor_mask)
        else:
            batch_size = video_data.size(0) if video_data is not None else sensor_data.size(0) if sensor_data is not None else integral_data.size(0)
            sensor_feat = self.sensor_null_token.expand(batch_size, -1)
        

        if use_integral and integral_data is not None:
            integral_mask = (integral_data.abs().sum(dim=-1) == 0)
            integral_feat = self.integral_encoder(integral_data, integral_mask)
        else:
            batch_size = video_data.size(0) if video_data is not None else sensor_data.size(0) if sensor_data is not None else integral_data.size(0)
            integral_feat = self.integral_null_token.expand(batch_size, -1)
        

        if use_video and video_data is not None:
            video_feat = self.video_encoder(video_data, video_lengths)
        else:
            batch_size = sensor_data.size(0) if sensor_data is not None else integral_data.size(0)
            video_feat = self.video_null_token.expand(batch_size, -1)
        

        fused_feat = self.cross_attention(sensor_feat, integral_feat, video_feat)
        
        if return_features:
            return fused_feat
        

        output = self.classifier(fused_feat)
        
        return output
    
    def load_pretrained_base_model(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)

        print("Loaded pretrained weights (null tokens initialized randomly)")


class ModelTester:
    
    def __init__(self, model_path, device, save_dir='./test_results_student', use_modal_flags=True):

        self.device = device
        self.save_dir = save_dir
        self.use_modal_flags = use_modal_flags
        os.makedirs(save_dir, exist_ok=True)

        self.classes = [
            '1_ND_Normal Driving',
            '2_OHD_One-Handed Driving',
            '3_HOD_Hands-Off Driving',
            '4_CA_Calling',
            '5_UPOH_Using Phone with One Hand',
            '6_UPBH_Using Phone with Both Hands',
            '7_DW_Drinking Water',
            '8_SM_Smoking',
            '9_OCC_Operating Central Console',
            '10_TWP_Talking with Passenger'
        ]

        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        print(f"Loading model from {model_path}...")

        if self.use_modal_flags:
            print("Using model with modal flags support")
            model = OptimizedStudentModelWithModalFlags(num_classes=10).to(self.device)
        else:
            model = OptimizedStudentModel(num_classes=10).to(self.device)
        

        if model_path.endswith('.pt'):
            loaded_model = torch.load(model_path, map_location=self.device)

            if self.use_modal_flags and not isinstance(loaded_model, OptimizedStudentModelWithModalFlags):
                print("Converting old model to modal flags version...")
                old_state_dict = loaded_model.state_dict()
                model.load_pretrained_base_model(old_state_dict)
            else:
                model = loaded_model
        else:

            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            if self.use_modal_flags:
                model.load_pretrained_base_model(state_dict)
            else:
                model.load_state_dict(state_dict)
        
        print("Model loaded successfully!")
        return model
    
    def calculate_model_info(self):

        results = {}
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        results['total_parameters'] = total_params
        results['trainable_parameters'] = trainable_params
        results['total_parameters_M'] = total_params / 1e6
        results['trainable_parameters_M'] = trainable_params / 1e6

        model_size = 0
        for param in self.model.parameters():
            model_size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            model_size += buffer.nelement() * buffer.element_size()
        
        results['model_size_MB'] = model_size / (1024 * 1024)

        batch_size = 1
        sensor_data = torch.randn(batch_size, 100, 10).to(self.device)
        integral_data = torch.randn(batch_size, 100, 10).to(self.device)
        video_data = torch.randn(batch_size, 30, 3, 224, 224).to(self.device)  # 30帧
        video_lengths = torch.tensor([30]).to(self.device)

        for _ in range(10):
            with torch.no_grad():
                if self.use_modal_flags:
                    _ = self.model(sensor_data, integral_data, video_data, video_lengths,
                                 use_sensor=True, use_integral=True, use_video=True)
                else:
                    _ = self.model(sensor_data, integral_data, video_data, video_lengths)

        num_runs = 100
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                if self.use_modal_flags:
                    _ = self.model(sensor_data, integral_data, video_data, video_lengths,
                                 use_sensor=True, use_integral=True, use_video=True)
                else:
                    _ = self.model(sensor_data, integral_data, video_data, video_lengths)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs
        results['avg_inference_time_ms'] = avg_inference_time * 1000

        info_path = os.path.join(self.save_dir, 'model_size.txt')
        
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"total_parameters: {results['total_parameters']:,} ({results['total_parameters_M']:.2f}M)\n")
            f.write(f"trainable_parameters: {results['trainable_parameters']:,} ({results['trainable_parameters_M']:.2f}M)\n")
            f.write(f"\nmodel_size：\n")
            f.write(f"{results['model_size_MB']:.2f} MB\n")
            f.write(f"\navg_inference_time：\n")
            f.write(f"{results['avg_inference_time_ms']:.2f} ms\n")

        
        print(f"\nModel Information saved to model_size.txt")
        return results
    
    def _save_metrics_to_txt(self, cm, metrics, filename):
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("confusion matrix：\n")
            f.write("-" * 100 + "\n")

            f.write("truth\\prediction\t")
            for i in range(len(self.classes)):
                f.write(f"class{i+1}\t")
            f.write("\n")
            

            for i in range(len(self.classes)):
                f.write(f"class{i+1}\t\t")
                for j in range(len(self.classes)):
                    f.write(f"{cm[i, j]}\t")
                f.write("\n")
            
            f.write("\n" + "-" * 100 + "\n\n")
            

            f.write("class-label：\n")
            for i, cls in enumerate(self.classes):
                f.write(f"class{i+1}: {cls}\n")
            
            f.write("\n" + "-" * 100 + "\n\n")

            f.write("evaluation：\n")
            f.write(f"mAcc : {metrics['mAcc']:.4f}\n")
            f.write(f"mP : {metrics['mP']:.4f}\n")
            f.write(f"mR: {metrics['mR']:.4f}\n")
            f.write(f"mF1 : {metrics['mF1']:.4f}\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            
            f.write("\n" + "-" * 100 + "\n\n")

            f.write("each class details：\n")
            for i, cls in enumerate(self.classes):
                f.write(f"\n{cls}:\n")
                f.write(f"  per_class_accuracy: {metrics['per_class_accuracy'][i]:.4f}\n")
                f.write(f"  per_class_precision: {metrics['per_class_precision'][i]:.4f}\n")
                f.write(f"  per_class_recall: {metrics['per_class_recall'][i]:.4f}\n")
                f.write(f"  per_class_f1: {metrics['per_class_f1'][i]:.4f}\n")
        
        print(f"Results saved to {filename}")
    
    def evaluate_test_set(self, test_loader, save_filename='test_avg.txt'):
        print("\nEvaluating on testing set...")
        
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (sensor, integral, images, labels, video_lengths) in enumerate(test_loader):
                sensor = sensor.to(self.device)
                integral = integral.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                video_lengths = video_lengths.to(self.device)

                if self.use_modal_flags:
                    outputs = self.model(sensor, integral, images, video_lengths,
                                       use_sensor=True, use_integral=True, use_video=True)
                else:
                    outputs = self.model(sensor, integral, images, video_lengths)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx}/{len(test_loader)} batches")

        cm = confusion_matrix(all_labels, all_predictions)
        metrics = self._calculate_metrics(cm, all_labels, all_predictions)

        self._save_metrics_to_txt(cm, metrics, save_filename)
        
        return cm, metrics
    
    def _calculate_metrics(self, cm, true_labels, pred_labels):

        per_class_accuracy = []
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        
        for i in range(len(self.classes)):

            class_mask = np.array(true_labels) == i
            if class_mask.sum() > 0:
                class_acc = (np.array(pred_labels)[class_mask] == i).mean()
                per_class_accuracy.append(class_acc)
            else:
                per_class_accuracy.append(0)
            

            class_precision = precision_score(true_labels, pred_labels, labels=[i], average=None, zero_division=0)
            class_recall = recall_score(true_labels, pred_labels, labels=[i], average=None, zero_division=0)
            class_f1 = f1_score(true_labels, pred_labels, labels=[i], average=None, zero_division=0)
            
            per_class_precision.append(class_precision[0] if len(class_precision) > 0 else 0)
            per_class_recall.append(class_recall[0] if len(class_recall) > 0 else 0)
            per_class_f1.append(class_f1[0] if len(class_f1) > 0 else 0)
        

        metrics = {
            'mAcc': np.mean(per_class_accuracy),
            'mP': np.mean(per_class_precision),
            'mR': np.mean(per_class_recall),
            'mF1': np.mean(per_class_f1),
            'per_class_accuracy': per_class_accuracy,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'overall_accuracy': accuracy_score(true_labels, pred_labels)
        }
        
        return metrics


def run_complete_test(model_path, data_root='./data_3', batch_size=4, use_modal_flags=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    tester = ModelTester(model_path, device, 
                        save_dir='./test_results_student',
                        use_modal_flags=use_modal_flags)

    print("\n" + "=" * 50)
    print("Calculating model information...")
    model_info = tester.calculate_model_info()
    print(f"Model Parameters: {model_info['total_parameters_M']:.2f}M")
    print(f"Model Size: {model_info['model_size_MB']:.2f}MB")
    print(f"Inference Time: {model_info['avg_inference_time_ms']:.2f}ms")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    print("\n" + "=" * 50)
    print("Loading test dataset...")
    test_dataset = DrivingBehaviorDataset(
        root_dir=data_root,
        mode='testing',
        transform=transform,
        augment=False
    )
    print(f"Test samples loaded: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print("\n" + "=" * 50)
    print("Test: Full modality test (Video + Sensor + Integral)...")
    cm_full, metrics_full = tester.evaluate_test_set(test_loader, 'test_video_sensor_integral.txt')
    print(f"Full Test - mAcc: {metrics_full['mAcc']:.4f}, mF1: {metrics_full['mF1']:.4f}")



if __name__ == "__main__":

    model_path = './checkpoints_student_improved/best_model.pth'
    data_root = './data_3/'

    # use_modal_flags=True:
    run_complete_test(model_path, data_root, batch_size=2, use_modal_flags=True)