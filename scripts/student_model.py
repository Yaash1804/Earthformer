import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    A simple ConvLSTM Cell implementation.
    Based on the paper: "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # Combined convolution for all 4 gates (input, forget, output, cell)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.
        
        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (batch_size, input_dim, height, width)
        cur_state: tuple
            Tuple containing the previous hidden state (h_cur) and cell state (c_cur)
            h_cur, c_cur shape: (batch_size, hidden_dim, height, width)
            
        Returns
        -------
        h_next, c_next: torch.Tensor
            Next hidden state and cell state
        """
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (batch_size, input_dim + hidden_dim, H, W)
        
        # Apply combined convolution
        combined_conv = self.conv(combined)
        
        # Split the output into the 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        
        # Apply activations
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Calculate next cell state
        c_next = f * c_cur + i * g
        # Calculate next hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize hidden state and cell state with zeros.
        
        Parameters
        ----------
        batch_size: int
            Size of the batch
        image_size: (int, int)
            Height and width of the image
            
        Returns
        -------
        (h, c): tuple
            Tuple of zero-initialized hidden and cell states
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTMStudent(nn.Module):
    """
    The ConvLSTM Student model.
    Based on the architecture from the original report: 
    Two ConvLSTM layers with 12 filters and 3x3 kernels.
    """
    def __init__(self, input_dim=1, hidden_dim=12, kernel_size=(3, 3), num_layers=2, bias=True):
        """
        Initialize the ConvLSTM Student model.
        
        Parameters
        ----------
        input_dim: int
            Number of channels in input tensor (1 for SST)
        hidden_dim: int
            Number of hidden channels (12 as per original report)
        kernel_size: (int, int)
            Kernel size (3x3 as per original report)
        num_layers: int
            Number of ConvLSTM layers (2 as per original report)
        bias: bool
            Whether to add bias
        """
        super(ConvLSTMStudent, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                                hidden_dim=self.hidden_dim,
                                                kernel_size=kernel_size,
                                                bias=bias))
        
        # Final convolution layer to map hidden state to 1 output channel
        self.output_conv = nn.Conv2d(in_channels=self.hidden_dim,
                                       out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=0,
                                       bias=True)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass for the ConvLSTM Student.
        
        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (b, seq_len, c, h, w)
            For this project, seq_len=1 and c=1.
        hidden_state: list
            List of (h, c) tuples for each layer
            
        Returns
        -------
        output_tensor: torch.Tensor
            Output tensor of shape (b, 1, 1, h, w)
        """
        # --- FIX: Renamed h, w to H, W to avoid variable collision ---
        b, seq_len, _, H, W = input_tensor.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            # --- FIX: Use H, W ---
            hidden_state = self._init_hidden(batch_size=b, image_size=(H, W))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            # 'h' and 'c' here are TENSORS (the hidden states)
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Unroll in time (seq_len is 1 for this project)
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            
            # Stack the time-steps
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output # Output of this layer is input to next
            
            last_state_list.append([h, c])

        # We only care about the output of the last layer
        # layer_output shape is (b, seq_len, hidden_dim, H, W)
        
        # Apply output conv to each time step
        # (b, seq_len, hidden_dim, H, W) -> (b * seq_len, hidden_dim, H, W)
        # --- FIX: Use H, W ---
        output = layer_output.view(b * seq_len, self.hidden_dim, H, W)
        output = self.output_conv(output)
        
        # (b * seq_len, 1, H, W) -> (b, seq_len, 1, H, W)
        # --- FIX: Use H, W ---
        output = output.view(b, seq_len, 1, H, W)

        return output

    def _init_hidden(self, batch_size, image_size):
        """Initializes hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
