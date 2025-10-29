import torch

class CNN(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        channels=[32, 64, 128],
        kernel_size: int = 3,
        pool_kernel_size: int = 2,
        dropout_conv: float | list[float] = [0.2, 0.2, 0.2],
        dropout_classifier: float = 0.4,
        classifier_hidden: int = 512,
        input_size: int = 128,
    ):
        super().__init__()

        padding = kernel_size
        conv_blocks = []
        current_in = in_channels
        dropout_values = dropout_conv
        
        if len(channels) != len(dropout_values):
            raise ValueError("Length of dropout_conv must be the same as number of channels")
        
        for block_idx, out_channels in enumerate(channels):
            # the convolution
            conv_blocks.append(
                torch.nn.Conv2d(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )
            conv_blocks.append(torch.nn.ReLU(inplace=True)) # relu
            # dropout
            dropout_value = dropout_values[block_idx]
            if dropout_value > 0:
                conv_blocks.append(torch.nn.Dropout(p=dropout_value))
            conv_blocks.append(torch.nn.MaxPool2d(kernel_size=pool_kernel_size)) # maxpool2d
            current_in = out_channels

        self.features = torch.nn.Sequential(*conv_blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            flattened_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=flattened_dim, out_features=classifier_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=dropout_classifier),
            torch.nn.Linear(in_features=classifier_hidden, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# class CNN(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         in_channels: int = 3,
#         channels=(32, 64, 128),
#         kernel_size: int = 3,
#         pool_kernel_size: int = 2,
#         dropout_conv: float | Sequence[float] = (0.2, 0.0, 0.0),
#         dropout_classifier: float = 0.5,
#         classifier_hidden: int = 512,
#         input_size: int = 128,
#     ):
#         super().__init__()

#         padding = kernel_size // 2
#         conv_blocks = []
#         current_in = in_channels
#         if isinstance(dropout_conv, (list, tuple)):
#             dropout_values = list(dropout_conv)
#         elif isinstance(dropout_conv, Sequence) and not isinstance(dropout_conv, (str, bytes, bytearray)):
#             dropout_values = list(dropout_conv)
#         else:
#             dropout_values = [float(dropout_conv)] * len(channels)

#         if len(dropout_values) != len(channels):
#             raise ValueError("Length of 'dropout_conv' must match the number of convolutional blocks.")

#         for block_idx, out_channels in enumerate(channels):
#             conv_blocks.append(
#                 nn.Conv2d(
#                     in_channels=current_in,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     padding=padding,
#                 )
#             )
#             conv_blocks.append(nn.ReLU(inplace=True))
#             dropout_value = dropout_values[block_idx]
#             if dropout_value > 0:
#                 conv_blocks.append(nn.Dropout(p=dropout_value))
#             conv_blocks.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
#             current_in = out_channels

#         self.features = nn.Sequential(*conv_blocks)

#         with torch.no_grad():
#             dummy = torch.zeros(1, in_channels, input_size, input_size)
#             flattened_dim = self.features(dummy).view(1, -1).shape[1]

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=flattened_dim, out_features=classifier_hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout_classifier),
#             nn.Linear(in_features=classifier_hidden, out_features=num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         return self.classifier(x)
