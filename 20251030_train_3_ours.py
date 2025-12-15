import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import glob
import time
import json
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

# seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class DrivingBehaviorDataset(Dataset):
    """Dataset-Multi-mode-Argumentation"""

    def __init__(self, root_dir: str, mode: str = 'training', transform=None, augment=False):
        """
        Args:
            root_dir
            mode: 'training'  'testing'
            transform
            augment
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.augment = augment and (mode == 'training')  # only training

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

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

        # Normalization
        self.sensor_stats = None
        self.integral_stats = None
        if self.samples:
            print(f"\n{mode.capitalize()} dataset loading: total {len(self.samples)} samples")
            self._compute_data_statistics()
        else:
            print(f"\n warning: {mode} no sample")

        if self.augment:
            self.sensor_augmentation = SensorAugmentation()
            self.video_augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])

    def _compute_data_statistics(self):

        sensor_data_list = []
        integral_data_list = []

        sample_indices = np.random.choice(len(self.samples),
                                          min(100, len(self.samples)),
                                          replace=False)

        for idx in sample_indices:
            sample = self.samples[idx]
            try:
                sensor_df = pd.read_csv(sample['sensor_csv'])
                sensor_data = sensor_df.iloc[:, -10:].apply(pd.to_numeric, errors='coerce').values


                sensor_data = sensor_data[~np.isnan(sensor_data).any(axis=1)]
                sensor_data = sensor_data[~np.isinf(sensor_data).any(axis=1)]

                if len(sensor_data) > 0:
                    sensor_data_list.append(sensor_data)


                integral_df = pd.read_csv(sample['integral_csv'])
                integral_data = integral_df.iloc[:, -10:].apply(pd.to_numeric, errors='coerce').values


                integral_data = integral_data[~np.isnan(integral_data).any(axis=1)]
                integral_data = integral_data[~np.isinf(integral_data).any(axis=1)]

                if len(integral_data) > 0:
                    integral_data_list.append(integral_data)

            except Exception as e:
                print(f"calculate samples {idx} error: {e}")
                continue

        if sensor_data_list:
            all_sensor_data = np.vstack(sensor_data_list)
            self.sensor_stats = {
                'mean': np.mean(all_sensor_data, axis=0).astype(np.float32),
                'std': np.std(all_sensor_data, axis=0).astype(np.float32) + 1e-6
            }

        if integral_data_list:
            all_integral_data = np.vstack(integral_data_list)
            self.integral_stats = {
                'mean': np.mean(all_integral_data, axis=0).astype(np.float32),
                'std': np.std(all_integral_data, axis=0).astype(np.float32) + 1e-6
            }

    def _normalize_data(self, data, stats):
        if stats is None:
            data_mean = np.mean(data, axis=0, keepdims=True)
            data_std = np.std(data, axis=0, keepdims=True) + 1e-6
            return (data - data_mean) / data_std
        else:
            return (data - stats['mean']) / stats['std']

    def _load_samples(self) -> List[Dict]:
        samples = []
        base_path = os.path.join(self.root_dir, self.mode)

        for class_name in self.classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue

            for sample_folder in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_folder)
                if not os.path.isdir(sample_path):
                    continue

                sensor_csv = None
                integral_csv = None
                for file in os.listdir(sample_path):
                    if 'sensor' in file.lower() and file.endswith('.csv'):
                        sensor_csv = os.path.join(sample_path, file)
                    elif 'integral' in file.lower() and file.endswith('.csv'):
                        integral_csv = os.path.join(sample_path, file)

                images = sorted(glob.glob(os.path.join(sample_path, '*.jpg')))

                if sensor_csv and integral_csv and images:
                    samples.append({
                        'sensor_csv': sensor_csv,
                        'integral_csv': integral_csv,
                        'images': images,
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name,
                        'sample_folder': sample_folder
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]


        try:
            sensor_df = pd.read_csv(sample['sensor_csv'])
            sensor_data = sensor_df.iloc[:, -10:].apply(pd.to_numeric, errors='coerce').values

            MAX_SEQ_LEN = 1024
            if len(sensor_data) > MAX_SEQ_LEN:
                sensor_data = sensor_data[:MAX_SEQ_LEN]

            sensor_data = np.nan_to_num(sensor_data, nan=0.0, posinf=1e6, neginf=-1e6)
            sensor_data = self._normalize_data(sensor_data, self.sensor_stats)
            sensor_data = np.clip(sensor_data, -10, 10)
            sensor_data = torch.FloatTensor(sensor_data)

            if self.augment and self.sensor_augmentation:
                sensor_data = self.sensor_augmentation(sensor_data)

        except Exception as e:
            print(f"sensor_data_error: {e}")
            sensor_data = torch.zeros(100, 10)

        try:
            integral_df = pd.read_csv(sample['integral_csv'])
            integral_data = integral_df.iloc[:, -10:].apply(pd.to_numeric, errors='coerce').values

            if len(integral_data) > MAX_SEQ_LEN:
                integral_data = integral_data[:MAX_SEQ_LEN]

            integral_data = np.nan_to_num(integral_data, nan=0.0, posinf=1e6, neginf=-1e6)
            integral_data = self._normalize_data(integral_data, self.integral_stats)
            integral_data = np.clip(integral_data, -10, 10)
            integral_data = torch.FloatTensor(integral_data)

            if self.augment and self.sensor_augmentation:
                integral_data = self.sensor_augmentation(integral_data)

        except Exception as e:
            print(f"integral_data_error: {e}")
            integral_data = torch.zeros(100, 10)

        images = []
        for img_path in sample['images']:
            try:
                img = Image.open(img_path).convert('RGB')

                if self.augment and self.video_augmentation:
                    img = self.video_augmentation(img)

                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"image_data_error {img_path}: {e}")

        # To tensor
        if images:
            images = torch.stack(images)
        else:
            print(f"warning: sample_ {idx} has no image")
            images = torch.zeros(1, 3, 224, 224)

        return sensor_data, integral_data, images, sample['label']


class SensorAugmentation(nn.Module):

    def __init__(self):
        super().__init__()
        self.noise_std = 0.1
        self.scale_range = (0.8, 1.2)
        self.training = True

    def forward(self, x):
        if not self.training:
            return x

        if random.random() < 0.3:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        if random.random() < 0.3:
            scale = random.uniform(*self.scale_range)
            x = x * scale

        if random.random() < 0.2 and x.size(0) > 10:
            seq_len = x.size(0)
            warp_point = random.randint(seq_len // 4, 3 * seq_len // 4)
            speed = random.uniform(0.9, 1.1)

            if speed != 1.0:
                indices = torch.arange(seq_len, dtype=torch.float32)
                indices[:warp_point] *= speed
                indices[warp_point:] = indices[warp_point:] * speed + (1 - speed) * warp_point
                indices = torch.clamp(indices, 0, seq_len - 1).long()
                x = x[indices]

        return x


class TemporalShift(nn.Module):

    def __init__(self, n_div=8):
        super().__init__()
        self.n_div = n_div

    def forward(self, x):
        # x - (B*T, C, H, W)
        if x.dim() != 4:
            return x

        BT, C, H, W = x.shape

        fold = C // self.n_div

        if fold == 0:
            return x
        out = x.clone()

        if BT > 1:
            out[1:, :fold] = x[:-1, :fold]  # shift forward
            if fold * 2 <= C:
                out[:-1, fold:2 * fold] = x[1:, fold:2 * fold]  # shift backward

        return out


class EfficientVideoEncoder(nn.Module):
    """MobileNetV3 + Temporal Shift Module"""

    def __init__(self, hidden_dim=256):
        super().__init__()

        # MobileNetV3-Large-backbone
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.features = mobilenet.features

        # Temporal Shift Module
        self.temporal_shift = TemporalShift(n_div=8)


        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.features(dummy_input)
            self.feat_dim = dummy_output.shape[1]
            print(f"MobileNetV3 feature dimension: {self.feat_dim}")

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.output_dim = hidden_dim

    def forward(self, x, lengths=None):
        # x: (batch, num_frames, 3, H, W)
        B, T, C, H, W = x.shape

        # (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        x = self.temporal_shift(x)

        x = self.features(x)  # (B*T, channels, h, w)

        _, channels, h, w = x.shape

        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B*T, channels, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B*T, channels)

        x = x.view(B, T, channels)  # (B, T, channels)

        if channels != self.feat_dim:
            x = F.linear(x, torch.eye(self.feat_dim, channels, device=x.device))


        if lengths is not None:

            mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x_sum = x.sum(dim=1)
            x_mean = x_sum / lengths.unsqueeze(1).float().clamp(min=1)
        else:
            x_mean = x.mean(dim=1)  # (B, feat_dim)

        output = self.fc(x_mean)

        return output


class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LocalAttention(nn.Module):

    def __init__(self, dim, window_size=32, n_heads=4, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0

        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        B, L, D = x.shape

        # Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, L, head_dim)

        if L <= self.window_size:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        else:
            n_windows = (L + self.window_size - 1) // self.window_size

            # Padding to make L divisible by window_size
            pad_len = n_windows * self.window_size - L
            if pad_len > 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))

            q = q.reshape(B, self.n_heads, n_windows, self.window_size, self.head_dim)
            k = k.reshape(B, self.n_heads, n_windows, self.window_size, self.head_dim)
            v = v.reshape(B, self.n_heads, n_windows, self.window_size, self.head_dim)

            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, n_windows, window_size, window_size)
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            out = attn @ v  # (B, n_heads, n_windows, window_size, head_dim)
            out = out.reshape(B, self.n_heads, n_windows * self.window_size, self.head_dim)
            out = out.transpose(1, 2).reshape(B, n_windows * self.window_size, D)

            # remove padding
            if pad_len > 0:
                out = out[:, :L]

        out = self.proj(out)
        out = self.dropout(out)
        out = self.norm(out + x)

        return out


class LightweightSensorEncoder(nn.Module):

    def __init__(self, input_dim=10, d_model=64, n_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.conv_blocks = nn.Sequential(
            DepthwiseSeparableConv1d(d_model, d_model, kernel_size=7, padding=3),
            DepthwiseSeparableConv1d(d_model, d_model, kernel_size=5, padding=2),
            DepthwiseSeparableConv1d(d_model, d_model, kernel_size=3, padding=1),
        )

        self.attention_layers = nn.ModuleList([
            LocalAttention(dim=d_model, window_size=32, n_heads=4)
            for _ in range(n_layers)
        ])

        self.output_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = d_model

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        B, L, _ = x.shape

        x = self.input_proj(x)  # (B, L, d_model)

        x_conv = x.transpose(1, 2)  # (B, d_model, L)
        x_conv = self.conv_blocks(x_conv)
        x = x + x_conv.transpose(1, 2)  # 残差连接

        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / lengths.clamp(min=1)
        else:
            x = x.transpose(1, 2)  # (B, d_model, L)
            x = self.output_pool(x).squeeze(-1)  # (B, d_model)

        return x


class EfficientCrossModalAttention(nn.Module):

    def __init__(self, sensor_dim, video_dim, hidden_dim=128):
        super().__init__()

        self.sensor_proj = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.integral_proj = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.output_dim = hidden_dim

        self.attention_weights = None

    def forward(self, sensor_feat, integral_feat, video_feat):
        s_feat = self.sensor_proj(sensor_feat)  # (B, hidden_dim//2)
        i_feat = self.integral_proj(integral_feat)  # (B, hidden_dim//2)
        v_feat = self.video_proj(video_feat)  # (B, hidden_dim)

        sensor_combined = torch.cat([s_feat, i_feat], dim=-1)  # (B, hidden_dim)

        gate_input = torch.cat([sensor_combined, v_feat], dim=-1)  # (B, hidden_dim*2)
        gate_weight = self.gate(gate_input)  # (B, hidden_dim)

        self.attention_weights = gate_weight.detach()

        fused = torch.cat([
            sensor_combined * gate_weight,
            v_feat * (1 - gate_weight)
        ], dim=-1)  # (B, hidden_dim*2)

        output = self.output_proj(fused)

        return output


class OptimizedStudentModel(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.sensor_encoder = LightweightSensorEncoder(input_dim=10, d_model=64, n_layers=2)
        self.integral_encoder = LightweightSensorEncoder(input_dim=10, d_model=64, n_layers=2)

        self.video_encoder = EfficientVideoEncoder(hidden_dim=128)

        self.cross_attention = EfficientCrossModalAttention(
            sensor_dim=64,
            video_dim=128,
            hidden_dim=128
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

        self.aux_classifier = nn.Linear(128, num_classes)

    def forward(self, sensor_data, integral_data, video_data, video_lengths=None, return_features=False):
        # padding masks
        sensor_mask = (sensor_data.abs().sum(dim=-1) == 0)
        integral_mask = (integral_data.abs().sum(dim=-1) == 0)

        sensor_feat = self.sensor_encoder(sensor_data, sensor_mask)
        integral_feat = self.integral_encoder(integral_data, integral_mask)
        video_feat = self.video_encoder(video_data, video_lengths)

        fused_feat = self.cross_attention(sensor_feat, integral_feat, video_feat)

        if return_features:
            return fused_feat

        output = self.classifier(fused_feat)

        return output

    def get_attention_weights(self):

        return self.cross_attention.attention_weights



class DualTeacherDistillation:

    def __init__(self, teacher1_path, teacher2_path, device, temperature=4.0, alpha=0.7):
        """
        Args:
            teacher1_path: ResNet50+Transformer
            teacher2_path: ResNet18+Linformer
            device
            temperature
            alpha
        """
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.teacher1 = None
        self.teacher2 = None
        self.active_teacher = None
        self.single_teacher_mode = False

        print("loading (ResNet50+BiLSTM+Transformer+CMA)...")
        self.teacher1 = self.load_teacher_model(teacher1_path, 'resnet50')
        print("loading (ResNet18+BiLSTM+Linformer+CMA)...")
        self.teacher2 = self.load_teacher_model(teacher2_path, 'resnet18')

        if self.teacher1 is not None and self.teacher2 is not None:
            print("DT-MODE")
            self.single_teacher_mode = False
            self.active_teacher = None
        elif self.teacher1 is not None and self.teacher2 is None:
            print("ONLY TEACHER1 SINGLE-MODE")
            self.single_teacher_mode = True
            self.active_teacher = self.teacher1
        elif self.teacher1 is None and self.teacher2 is not None:
            print("ONLY TEACHER2 SINGLE-MODE")
            self.single_teacher_mode = True
            self.active_teacher = self.teacher2
        else:
            print("NO TEACHER NO KD")
            self.single_teacher_mode = True
            self.active_teacher = None


        if self.teacher1 is not None:
            self.teacher1.eval()
            for param in self.teacher1.parameters():
                param.requires_grad = False
            print("evaluation-teacher1")

        if self.teacher2 is not None:
            self.teacher2.eval()
            for param in self.teacher2.parameters():
                param.requires_grad = False
            print("evaluation-teacher2")

    def load_teacher_model(self, checkpoint_path, model_type):

        try:
            if model_type == 'resnet50':
                # ResNet50 + BiLSTM + Transformer + CMA
                from train_1_all_ggrecog_unfreeze import DrivingBehaviorClassifier as Teacher1Model
                model = Teacher1Model(num_classes=10, unfreeze_all=False)
                print(" ResNet50 + BiLSTM + Transformer + CMA")
            else:
                # ResNet18 + BiLSTM + Linformer + CMA
                from train_2_resnet18_linformer import DrivingBehaviorClassifier as Teacher2Model
                model = Teacher2Model(num_classes=10, unfreeze_all=False)
                print("ResNet18 + BiLSTM + Linformer + CMA")

            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"loading success: {checkpoint_path}")
            else:
                print(f"can not find teacher {checkpoint_path}，random begin")

            return model.to(self.device)

        except ImportError as e:
            print(f"error: can not lead into teacher - {e}")
            return None
        except Exception as e:
            print(f"error: loading teacher: {e}")
            return None

    def distillation_loss(self, student_logits, labels, sensor_data, integral_data,
                          video_data, video_lengths):

        ce_loss = F.cross_entropy(student_logits, labels)


        if self.single_teacher_mode and self.active_teacher is None:

            return ce_loss, ce_loss.item(), 0.0


        with torch.no_grad():
            if self.single_teacher_mode and self.active_teacher is not None:
                teacher_logits = self.active_teacher(sensor_data, integral_data, video_data, video_lengths)
            elif not self.single_teacher_mode and self.teacher1 is not None and self.teacher2 is not None:
                teacher1_logits = self.teacher1(sensor_data, integral_data, video_data, video_lengths)
                teacher2_logits = self.teacher2(sensor_data, integral_data, video_data, video_lengths)
                teacher_logits = (teacher1_logits + teacher2_logits) / 2
            else:
                return ce_loss, ce_loss.item(), 0.0

        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        return total_loss, ce_loss.item(), kl_loss.item()


class DistillationTrainer:

    def __init__(self, student_model, teacher1_path, teacher2_path, device,
                 save_dir='./checkpoints_student'):
        self.student = student_model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.distillation = DualTeacherDistillation(
            teacher1_path=teacher1_path,
            teacher2_path=teacher2_path,
            device=device,
            temperature=4.0,
            alpha=0.7
        )

        self.train_losses = []
        self.ce_losses = []
        self.kl_losses = []
        self.best_loss = float('inf')

        self.scaler = GradScaler()

        self.attention_dir = os.path.join(save_dir, 'attention_visualizations')
        os.makedirs(self.attention_dir, exist_ok=True)

    def save_loss_history(self):
        epoch_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'ce_loss': self.ce_losses,
            'kl_loss': self.kl_losses
        })
        epoch_df.to_csv(os.path.join(self.save_dir, 'epoch_losses.csv'), index=False)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('总损失')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epoch_df['epoch'], epoch_df['ce_loss'], label='CE Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('分类损失')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epoch_df['epoch'], epoch_df['kl_loss'], label='KL Loss', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('蒸馏损失')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'))
        plt.close()

    def visualize_attention(self, dataloader, epoch):

        self.student.eval()

        with torch.no_grad():

            sensor, integral, images, labels, video_lengths = next(iter(dataloader))
            sensor = sensor.to(self.device)
            integral = integral.to(self.device)
            images = images.to(self.device)
            video_lengths = video_lengths.to(self.device)


            _ = self.student(sensor, integral, images, video_lengths)


            attention_weights = self.student.get_attention_weights()

            if attention_weights is not None:

                gate_weights = attention_weights[0].cpu().numpy()

                plt.figure(figsize=(10, 6))
                plt.bar(range(len(gate_weights)), gate_weights)
                plt.xlabel('features')
                plt.ylabel('gate weights')
                plt.title(f'cross modal (Epoch {epoch})')
                plt.ylim([0, 1])
                plt.grid(True, alpha=0.3)

                save_path = os.path.join(self.attention_dir, f'gate_weights_epoch_{epoch}.png')
                plt.savefig(save_path)
                plt.close()

                print(f"save to: {save_path}")

        self.student.train()

    def train_epoch(self, train_loader, optimizer, epoch):

        self.student.train()
        total_loss = 0
        total_ce_loss = 0
        total_kl_loss = 0
        correct = 0
        total = 0

        for batch_idx, (sensor, integral, images, labels, video_lengths) in enumerate(train_loader):
            sensor = sensor.to(self.device)
            integral = integral.to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            video_lengths = video_lengths.to(self.device)

            optimizer.zero_grad()


            with autocast():

                student_logits = self.student(sensor, integral, images, video_lengths)

                loss, ce_loss, kl_loss = self.distillation.distillation_loss(
                    student_logits, labels, sensor, integral, images, video_lengths
                )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_ce_loss += ce_loss
            total_kl_loss += kl_loss

            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f} (CE: {ce_loss:.4f}, KL: {kl_loss:.4f}), '
                      f'Acc: {100. * correct / total:.2f}%')

        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        accuracy = 100. * correct / total

        self.train_losses.append(avg_loss)
        self.ce_losses.append(avg_ce_loss)
        self.kl_losses.append(avg_kl_loss)

        return avg_loss, accuracy

    def train(self, train_loader, num_epochs, learning_rate=1e-4, resume_from=None):

        print("\n" + "=" * 60)
        if self.distillation.single_teacher_mode and self.distillation.active_teacher is None:
            print("no kd")
        elif self.distillation.single_teacher_mode:
            print("single kd")
        else:
            print("dt")
        print("=" * 60 + "\n")


        optimizer = optim.AdamW(self.student.parameters(), lr=learning_rate, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            print(f"resume: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.student.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            self.train_losses = checkpoint.get('train_losses', [])
            self.ce_losses = checkpoint.get('ce_losses', [])
            self.kl_losses = checkpoint.get('kl_losses', [])
            print(f"从epoch {start_epoch} 恢复训练")

        print("\n training...")
        for epoch in range(start_epoch, num_epochs):
            print(f'\n{"=" * 50}')
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'{"=" * 50}')


            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch + 1)

            print(f'\n avg loss: {train_loss:.4f}, acc: {train_acc:.2f}%')
            if self.kl_losses and self.kl_losses[-1] > 0:
                print(f'CE loss: {self.ce_losses[-1]:.4f}, KL loss: {self.kl_losses[-1]:.4f}')


            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'train_losses': self.train_losses,
                'ce_losses': self.ce_losses,
                'kl_losses': self.kl_losses
            }


            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
                print("✓ best model")

            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
                print(f"✓  epoch_{epoch + 1}")

            torch.save(checkpoint, os.path.join(self.save_dir, 'latest_model.pth'))

            self.save_loss_history()

            if (epoch + 1) % 20 == 0:
                print("\n visualization weights...")
                self.visualize_attention(train_loader, epoch + 1)


            scheduler.step()
            print(f' lr: {scheduler.get_last_lr()[0]:.6f}')

        torch.save(self.student, os.path.join(self.save_dir, 'final_model.pt'))
        print('\n' + '=' * 60)
        print('训练完成!')
        print('=' * 60)


def collate_fn(batch):

    sensor_data_list, integral_data_list, images_list, labels = zip(*batch)

    max_sensor_len = max(data.size(0) for data in sensor_data_list)
    padded_sensor = torch.zeros(len(batch), max_sensor_len, 10)
    for i, data in enumerate(sensor_data_list):
        padded_sensor[i, :data.size(0)] = data

    max_integral_len = max(data.size(0) for data in integral_data_list)
    padded_integral = torch.zeros(len(batch), max_integral_len, 10)
    for i, data in enumerate(integral_data_list):
        padded_integral[i, :data.size(0)] = data

    video_lengths = torch.tensor([images.size(0) for images in images_list])
    max_frames = max(video_lengths)

    padded_images = torch.zeros(len(batch), max_frames, 3, 224, 224)
    for i, images in enumerate(images_list):
        actual_frames = images.size(0)
        padded_images[i, :actual_frames] = images
        if actual_frames < max_frames:
            padded_images[i, actual_frames:] = images[-1].unsqueeze(0).repeat(max_frames - actual_frames, 1, 1, 1)

    labels = torch.LongTensor(labels)

    return padded_sensor, padded_integral, padded_images, labels, video_lengths


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f' device: {device}')

    data_root = './data_3'
    teacher1_path = './checkpoints_resnet50_transformer/best_model.pth'  # ResNet50+BiLSTM+Transformer+CMA模型
    teacher2_path = './checkpoints_resnet18_linformer/best_model.pth'  # ResNet18+BiLSTM+Linformer+CMA模型
    save_dir = './checkpoints_student'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("loading training set...")
    train_dataset = DrivingBehaviorDataset(
        root_dir=data_root,
        mode='training',
        transform=transform,
        augment=True
    )
    print(f"training_total: {len(train_dataset)}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  #
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )

    student_model = OptimizedStudentModel(num_classes=10).to(device)

    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"total_params: {total_params:,}")
    print(f"trainable_params: {trainable_params:,}")

    teacher_available = False
    if not os.path.exists(teacher1_path):
        print(f"\nwarning: no teacher1: {teacher1_path}")
    else:
        teacher_available = True
        print(f"find teacher1: {teacher1_path}")

    if not os.path.exists(teacher2_path):
        print(f"warning: no teacher2: {teacher2_path}")
    else:
        teacher_available = True
        print(f"find teacher2: {teacher2_path}")

    if not teacher_available:
        print("\nnote：no kd")
        user_continue = input("no kd？(y/n): ").strip().lower()
        if user_continue != 'y':
            print("cancel")
            return

    trainer = DistillationTrainer(
        student_model=student_model,
        teacher1_path=teacher1_path,
        teacher2_path=teacher2_path,
        device=device,
        save_dir=save_dir
    )


    latest_checkpoint = os.path.join(save_dir, 'latest_model.pth')
    resume_from = None

    if os.path.exists(latest_checkpoint):
        print(f"\nlatest_checkpoint: {latest_checkpoint}")
        user_input = input("continue？(y/n): ").strip().lower()
        if user_input == 'y':
            resume_from = latest_checkpoint
            print("go on...")

    trainer.train(
        train_loader,
        num_epochs=100,
        learning_rate=5e-5,
        resume_from=resume_from
    )

    print("\ntraining over！")
    print(f"save to: {save_dir}")


    if trainer.train_losses:
        print(f"\ntrain loss: {trainer.train_losses[-1]:.4f}")
        print(f"ce loss: {trainer.ce_losses[-1]:.4f}")
        print(f"kl loss: {trainer.kl_losses[-1]:.4f}")
        print(f"best loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()