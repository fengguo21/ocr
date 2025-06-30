import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    """双向LSTM层"""
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class CRNN(nn.Module):
    """CRNN模型 - 用于文本识别的经典架构"""
    
    def __init__(self, img_h, nc, nclass, nh, n_rnn=2, leaky_relu=False):
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                          nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i),
                              nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                      nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                      nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        conv = conv.transpose(0, 1)  # [b, w, c]

        # rnn features
        output = self.rnn(conv)
        return output


class AttentionCRNN(nn.Module):
    """带注意力机制的CRNN模型"""
    
    def __init__(self, img_h, nc, nclass, nh):
        super(AttentionCRNN, self).__init__()
        self.crnn = CRNN(img_h, nc, nh, 2)
        self.attention = nn.MultiheadAttention(nh, 8, batch_first=True)
        self.classifier = nn.Linear(nh, nclass)
        
    def forward(self, input):
        # 获取CRNN特征
        features = self.crnn.cnn(input)
        b, c, h, w = features.size()
        features = features.squeeze(2).permute(0, 2, 1)  # [b, w, c]
        
        # 通过LSTM
        lstm_out = self.crnn.rnn[0](features)  # 第一层LSTM
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 第二层LSTM
        final_out = self.crnn.rnn[1](attn_out)
        
        return final_out 