import torch
import torch.nn.functional as F
from torch import nn

# BASIS NETWORK


class PointNetfeat(nn.Module):
    """Feature extraction network for the basis model."""

    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, 128, 1)
        self.conv42 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.dense1 = torch.nn.Linear(1024, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn41 = nn.BatchNorm1d(128)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        """Forward pass for feature extraction."""
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn6(self.dense1(x)))
        x = F.relu(self.bn7(self.dense2(x)))

        # Repeat features for each point
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), None, None


class PointNetBasis(nn.Module):
    """Basis network for learning shape correspondences."""

    def __init__(self, k=20, feature_transform=False):
        super(PointNetBasis, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=False,
            feature_transform=feature_transform,
        )
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """Forward pass for the basis network."""
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x, None, None


# DESC NETWORK


class PointNetfeatDesc(nn.Module):
    """Feature extraction network for the descriptor model."""

    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeatDesc, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, 128, 1)
        self.conv42 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.dense1 = torch.nn.Linear(1024, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn41 = nn.BatchNorm1d(128)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        """Forward pass for feature extraction."""
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn6(self.dense1(x)))
        x = F.relu(self.bn7(self.dense2(x)))

        # Repeat features for each point
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), None, None


class PointNetDesc(nn.Module):
    """Descriptor network for learning shape descriptors."""

    def __init__(self, k=40, feature_transform=False):
        super(PointNetDesc, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeatDesc(
            global_feat=False,
            feature_transform=feature_transform,
        )
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """Forward pass for the descriptor network."""
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x, None, None
