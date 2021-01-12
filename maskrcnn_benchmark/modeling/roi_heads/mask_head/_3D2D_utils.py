import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from maskrcnn_benchmark.structures.bounding_box import BoxList

def box_extractor(boxes, disp_image, num_pts):
	"""
	Convert to pseudo-lidar repr.
	"""
	side_pts = int(math.sqrt(num_pts))
	extracted_areas = []
	x1,y1,x2,y2 = boxes._split_into_xyxy()

	x1,y1,x2,y2 = np.squeeze(x1.cpu().numpy().astype(int)), np.squeeze(y1.cpu().numpy().astype(int)),\
					 np.squeeze(x2.cpu().numpy().astype(int)), np.squeeze(y2.cpu().numpy().astype(int))

	if len(x1.shape) == 0: ## For the no-box handle
		x1, x2, y1, y2 = np.array([10]), np.array([10]), np.array([10]), np.array([10])

	num_boxes = x1.shape[0]
	pc = torch.cuda.FloatTensor(num_boxes, 3, side_pts**2).fill_(0)

	for i in range(num_boxes):
		if ((x2[i]-x1[i])/side_pts ==0) or ((y2[i]-y1[i])/side_pts==0):
			c, r= torch.meshgrid([torch.Tensor(x1[i].repeat(side_pts)), torch.Tensor(y1[i].repeat(side_pts))])
		else:
			c, r= torch.meshgrid([torch.arange(x1[i], x2[i], (x2[i]-x1[i])/side_pts), torch.arange(y1[i],y2[i], (y2[i]-y1[i])/side_pts)])

		c = c.cuda()
		r = r.cuda()		
		d = disp_image[r.type('torch.cuda.LongTensor'), c.type('torch.LongTensor')]
		points = torch.stack([c,r,d])
		points = torch.reshape(points,(3,-1))
		pc[i] = points

	return pc, (x1,y1,x2,y2)


class InstanceSeg(nn.Module):
	"""
	Point-Net structure
	"""
	def __init__(self, num_pts):
		super(InstanceSeg,self).__init__()
		self._3Dc1 = torch.nn.Conv1d(3,64,1)
		self.conv2 = torch.nn.Conv1d(64,64,1)
		self.conv3 = torch.nn.Conv1d(64,64,1)
		self.conv4 = torch.nn.Conv1d(64,128,1)
		self.conv5 = torch.nn.Conv1d(128,1024,1)
		self.conv6 = nn.Conv1d(1088, 512, 1)
		self.conv7 = nn.Conv1d(512, 256, 1)
		self._3Db1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(1024)
		self.bn6 = nn.BatchNorm1d(512)
		self.bn7 = nn.BatchNorm1d(256)
		self.num_pts = num_pts
		self.max_pool = nn.MaxPool1d(num_pts)
		
	def forward(self,x):
		out = F.relu(self._3Db1(self._3Dc1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		point_features = out
		out = F.relu(self.bn3(self.conv3(out)))
		out = F.relu(self.bn4(self.conv4(out)))
		out = F.relu(self.bn5(self.conv5(out)))
		global_features = self.max_pool(out)
		global_features_repeated = global_features.repeat(1,1, self.num_pts)
		out = F.relu(self.bn6(self.conv6(torch.cat([point_features, global_features_repeated],1))))
		out = F.relu(self.bn7(self.conv7(out)))
		return out

class p25DSeg(nn.Module):
	"""
	Convolution branch.
	"""
	def __init__(self, num_pts):
		super(p25DSeg, self).__init__()
		self._3Dc1 = torch.nn.Conv2d(1,64,1)
		self.conv2 = torch.nn.Conv2d(64,64,1)
		self.conv3 = torch.nn.Conv2d(64,64,1)
		self.conv4 = torch.nn.Conv2d(64,128,1)
		self.conv5 = torch.nn.Conv2d(128,1024,1)
		self.conv6 = nn.Conv2d(1024, 512, 1)
		self.conv7 = nn.Conv2d(512, 256, 1)
		self._3Db1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(64)
		self.bn4 = nn.BatchNorm2d(128)
		self.bn5 = nn.BatchNorm2d(1024)
		self.bn6 = nn.BatchNorm2d(512)
		self.bn7 = nn.BatchNorm2d(256)
		self.num_pts = num_pts
	def forward(self,x):
		out = F.relu(self._3Db1(self._3Dc1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = F.relu(self.bn3(self.conv3(out)))
		out = F.relu(self.bn4(self.conv4(out)))
		out = F.relu(self.bn5(self.conv5(out)))
		out = F.relu(self.bn6(self.conv6(out)))
		out = F.relu(self.bn7(self.conv7(out)))
		return out
