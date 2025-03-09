import math
import torch
from torch import nn
import torch.nn.functional as F


class ObjectEncoder(nn.Module):
    def __init__(self, obj_in_dim, hidden_size, dropout_prob=0.1):
        """
        Initialization method for the ObjectEncoder class.

        Args:
            obj_in_dim (int): The input dimension of object features.
            hidden_size (int): The hidden size of the linear layers in the encoder.
            dropout_prob (float, optional): The probability of dropout. Default is 0.1.

        Attributes:
            linear_obj_feat_to_mmt_in (nn.Linear): Linear layer for transforming object features.
            linear_obj_bbox_to_mmt_in (nn.Linear): Linear layer for transforming object bounding box features.
            obj_feat_layer_norm (nn.LayerNorm): Layer normalization for object features.
            obj_bbox_layer_norm (nn.LayerNorm): Layer normalization for object bounding box features.
            dropout (nn.Dropout): Dropout layer with the specified probability.

        """
        super().__init__()

        # Linear layer for object features transformation
        self.linear_obj_feat_to_mmt_in = nn.Linear(obj_in_dim, hidden_size)

        # Linear layer for object bounding box transformation
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, hidden_size)

        # Layer normalization for object features
        self.obj_feat_layer_norm = nn.LayerNorm(hidden_size)

        # Layer normalization for object bounding box features
        self.obj_bbox_layer_norm = nn.LayerNorm(hidden_size)

        # Dropout layer with specified probability
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, obj_boxes, obj_features):
        """
        Forward pass of the ObjectEncoder.

        Args:
            obj_boxes (torch.Tensor): Bounding box features of objects.
            obj_features (torch.Tensor): Features of objects.

        Returns:
            torch.Tensor: Encoded object features after linear transformations, layer normalization, and dropout.
                          Shape: (batch_size, seq_length, hidden_size)

        """
        # Normalize object features
        obj_features = F.normalize(obj_features, dim=-1)

        # Transform and normalize object features
        obj_features = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(obj_features))

        # Transform and normalize object bounding box features
        obj_bbox_features = self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_boxes))

        # Combine object features and object bounding box features with dropout
        return self.dropout(obj_features + obj_bbox_features)

class OCREncoder(nn.Module):
    """
    OCREncoder is a PyTorch module for encoding Optical Character Recognition (OCR) features.

    Args:
        ocr_in_dim (int): The input dimension of OCR features.
        hidden_size (int): The hidden size of the linear layers in the encoder.
        dropout_prob (float, optional): The probability of dropout. Default is 0.1.

    Attributes:
        linear_ocr_feat_to_mmt_in (nn.Linear): Linear layer for transforming combined OCR features.
        linear_ocr_bbox_to_mmt_in (nn.Linear): Linear layer for transforming OCR bounding box features.
        ocr_feat_layer_norm (nn.LayerNorm): Layer normalization for combined OCR features.
        ocr_bbox_layer_norm (nn.LayerNorm): Layer normalization for OCR bounding box features.
        dropout (nn.Dropout): Dropout layer with the specified probability.

    Methods:
        forward(ocr_boxes, ocr_token_embeddings, ocr_rec_features, ocr_det_features):
            Performs a forward pass through the OCREncoder.

            Args:
                ocr_boxes (torch.Tensor): OCR bounding box features.
                ocr_token_embeddings (torch.Tensor): OCR token embeddings.
                ocr_rec_features (torch.Tensor): OCR recognition features.
                ocr_det_features (torch.Tensor): OCR detection features.

            Returns:
                torch.Tensor: Encoded OCR features after linear transformations, layer normalization, and dropout.
                             Shape: (batch_size, seq_length, hidden_size)
    """
    def __init__(self, ocr_in_dim, hidden_size, dropout_prob=0.1):
        super().__init__()

        # Linear layer for combined OCR features transformation
        self.linear_ocr_feat_to_mmt_in = nn.Linear(ocr_in_dim, hidden_size)

        # Linear layer for OCR bounding box transformation
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, hidden_size)

        # Layer normalization for combined OCR features
        self.ocr_feat_layer_norm = nn.LayerNorm(hidden_size)

        # Layer normalization for OCR bounding box features
        self.ocr_bbox_layer_norm = nn.LayerNorm(hidden_size)

        # Dropout layer with specified probability
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, ocr_boxes, ocr_token_embeddings, ocr_rec_features, ocr_det_features):
        """
        Performs a forward pass through the OCREncoder.

        Args:
            ocr_boxes (torch.Tensor): OCR bounding box features.
            ocr_token_embeddings (torch.Tensor): OCR token embeddings.
            ocr_rec_features (torch.Tensor): OCR recognition features.
            ocr_det_features (torch.Tensor): OCR detection features.

        Returns:
            torch.Tensor: Encoded OCR features after linear transformations, layer normalization, and dropout.
                         Shape: (batch_size, seq_length, hidden_size)
        """
        # Normalize input features
        ocr_token_embeddings = F.normalize(ocr_token_embeddings, dim=-1)
        ocr_rec_features = F.normalize(ocr_rec_features, dim=-1)
        ocr_det_features = F.normalize(ocr_det_features, dim=-1)

        # Combine OCR features
        ocr_combine_features = torch.cat([ocr_token_embeddings, ocr_rec_features, ocr_det_features], dim=-1)

        # Transform and normalize combined OCR features
        ocr_combine_features = self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_combine_features))

        # Transform and normalize OCR bounding box features
        ocr_bbox_features = self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_boxes))

        # Combine transformed features with dropout
        return self.dropout(ocr_combine_features + ocr_bbox_features)

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding is a module that adds positional encoding to the input sequence.

    Args:
        d_model (int): The dimension of the input features.
        max_len (int, optional): The maximum length of the input sequence. Default is 512.
        dropout_prob (float, optional): The probability of dropout. Default is 0.1.

    Attributes:
        dropout (nn.Dropout): Dropout layer with the specified probability.
        pe (torch.Tensor): Positional encoding tensor of shape (1, max_len, d_model).

    Methods:
        forward(x):
            Adds positional encoding to the input sequence.

            Args:
                x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).

            Returns:
                torch.Tensor: Sequence with added positional encoding of the same shape as input.

    """

    def __init__(self, d_model, max_len=512, dropout_prob=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)

        position_ids = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(size=(1, max_len, d_model))
        pe[0, :, 0::2] = torch.sin(position_ids / div_term)
        pe[0, :, 1::2] = torch.cos(position_ids / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input sequence.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Sequence with added positional encoding of the same shape as input.
        """
        # x shape (batch_size, seq_length, d_model)
        return x + self.pe[:, :x.size(1), :]


class PrevPredEmbeddings(nn.Module):
    """
    PrevPredEmbeddings is a module for creating embeddings for answer and OCR tokens with positional encoding.

    Args:
        hidden_size (int): The hidden size of the embeddings.
        ln_eps (float, optional): Epsilon value for LayerNorm. Default is 1e-12.
        dropout_prob (float, optional): The probability of dropout. Default is 0.1.

    Attributes:
        position_embeddings (PositionalEncoding): Positional encoding for token embeddings.
        token_type_embeddings (nn.Embedding): Token type embeddings for distinguishing between answer and OCR tokens.
        ans_layer_norm (nn.LayerNorm): Layer normalization for answer embeddings.
        ocr_layer_norm (nn.LayerNorm): Layer normalization for OCR embeddings.
        emb_layer_norm (nn.LayerNorm): Layer normalization for final embeddings.
        emb_dropout (nn.Dropout): Dropout layer with the specified probability.

    Methods:
        forward(ans_emb, ocr_emb, labels):
            Generates embeddings for answer and OCR tokens with positional encoding.

            Args:
                ans_emb (torch.Tensor): Answer token embeddings.
                ocr_emb (torch.Tensor): OCR token embeddings.
                labels (torch.Tensor): Token type labels (0 for answer tokens, 1 for OCR tokens).

            Returns:
                torch.Tensor: Final embeddings after layer normalization and dropout.
                             Shape: (batch_size, seq_length, hidden_size).

    """

    def __init__(self, hidden_size, ln_eps=1e-12, dropout_prob=0.1):
        super().__init__()

        self.position_embeddings = PositionalEncoding(hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.token_type_embeddings.weight.data = nn.Parameter(torch.cat([torch.zeros(1, hidden_size),
                                                                         torch.ones(1, hidden_size)]))

        self.ans_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(dropout_prob)

    def forward(self, ans_emb, ocr_emb, labels):
        """
        Generates embeddings for answer and OCR tokens with positional encoding.

        Args:
            ans_emb (torch.Tensor): Answer token embeddings.
            ocr_emb (torch.Tensor): OCR token embeddings.
            labels (torch.Tensor): Token type labels (0 for answer tokens, 1 for OCR tokens).

        Returns:
            torch.Tensor: Final embeddings after layer normalization and dropout.
                         Shape: (batch_size, seq_length, hidden_size).
        """
        batch_size = labels.size(0)
        seq_length = labels.size(1)
        ans_num = ans_emb.size(0)

        # Apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)

        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_embeddings = self.token_type_embeddings(labels.ge(ans_num).long())  # N, T, hidden_size
        embeddings = self.emb_dropout(self.emb_layer_norm(self.position_embeddings(token_type_embeddings)))

        return _batch_gather(torch.cat([ans_emb.unsqueeze(0).expand(batch_size, -1, -1), ocr_emb], dim=1),
                             labels) + embeddings
    

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is a module that performs multi-head attention on input sequences.

    Args:
        d_model (int): The dimension of the input features.
        n_heads (int): The number of attention heads.
        d_k (int): The dimension of keys and queries in each attention head.
        causal (bool, optional): If True, applies causal masking to prevent attending to future positions.
                                 Default is False.

    Attributes:
        n_heads (int): The number of attention heads.
        d_k (int): The dimension of keys and queries in each attention head.
        query (nn.Linear): Linear layer for computing queries.
        key (nn.Linear): Linear layer for computing keys.
        value (nn.Linear): Linear layer for computing values.
        fc (nn.Linear): Linear layer for the final output after attention.
        causal (bool): If True, applies causal masking.

    Methods:
        forward(x, dec_size, attention_mask=None):
            Performs multi-head attention on the input sequence.

            Args:
                x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).
                dec_size (int): Size of the decoder sequence (number of tokens to attend to).
                attention_mask (torch.Tensor, optional): Mask to avoid attending to certain positions.
                                                          Default is None.

            Returns:
                torch.Tensor: Output after multi-head attention of shape (batch_size, seq_length, d_model).

    """

    def __init__(self, d_model, n_heads, d_k, causal=False):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k

        # Linear layers for queries, keys, and values
        self.query = nn.Linear(d_model, n_heads * d_k)
        self.key = nn.Linear(d_model, n_heads * d_k)
        self.value = nn.Linear(d_model, n_heads * d_k)

        # Linear layer for the final output after attention
        self.fc = nn.Linear(n_heads * d_k, d_model)
        self.causal = causal

    def forward(self, x, dec_size, attention_mask=None):
        """
        Performs multi-head attention on the input sequence.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).
            dec_size (int): Size of the decoder sequence (number of tokens to attend to).
            attention_mask (torch.Tensor, optional): Mask to avoid attending to certain positions.
                                                      Default is None.

        Returns:
            torch.Tensor: Output after multi-head attention of shape (batch_size, seq_length, d_model).
        """
        # x shape (batch_size, seq_length, d_model)

        N = x.size(0)
        T = x.size(1)

        # Pass through linear to get q, k, v
        q = self.query(x).view(N, T, self.n_heads, self.d_k).transpose(1, 2) # batch_size, n_heads, T, d_k
        k = self.key(x).view(N, T, self.n_heads, self.d_k).transpose(1, 2) # batch_size, n_heads, T, d_k
        v = self.query(x).view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        # Get the attention scores
        attn_scores = (q @ k.mT) / math.sqrt(self.d_k) # batch_size, n_heads, T_dec, T_enc

        # Mask the padding values and (if causal)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))

        if self.causal:
            causal_mask = torch.tril(torch.ones(dec_size, dec_size))
            extend_causal_mask = torch.ones((T, T))
            extend_causal_mask[:, -dec_size:] = torch.cat([torch.zeros((T - dec_size, dec_size)), causal_mask])
            extend_causal_mask[-dec_size:, -dec_size:] = causal_mask
            extend_causal_mask = extend_causal_mask.to(attn_scores.device)

            attn_scores = attn_scores.masked_fill(extend_causal_mask[None, None, :, :]==0, float('-inf'))

        # Get the attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Get the values
        A = attn_weights @ v # batch_size, n_heads, T_dec, d_k

        # Reshape to batch_size, T_dec, n_heads * d_k
        A = A.transpose(1, 2).contiguous().view(N, T, self.n_heads * self.d_k)

        return self.fc(A)

class DecoderBlock(nn.Module):
    """
    DecoderBlock is a module representing a single block in a Transformer decoder.

    Args:
        d_model (int): The dimension of the input features.
        n_heads (int): The number of attention heads in the MultiHeadAttention layer.
        d_k (int): The dimension of keys and queries in each attention head.
        dropout_prob (float, optional): The probability of dropout. Default is 0.1.

    Attributes:
        mha (MultiHeadAttention): MultiHeadAttention layer for self-attention.
        ln1 (nn.LayerNorm): Layer normalization after the first sub-layer.
        ln2 (nn.LayerNorm): Layer normalization after the second sub-layer.
        ffn (nn.Sequential): Feedforward neural network with GELU activation for the second sub-layer.
        dropout (nn.Dropout): Dropout layer with the specified probability.

    Methods:
        forward(x, dec_size, attention_mask=None):
            Performs the forward pass through the DecoderBlock.

            Args:
                x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).
                dec_size (int): Size of the decoder sequence (number of tokens to attend to).
                attention_mask (torch.Tensor, optional): Mask to avoid attending to certain positions.
                                                          Default is None.

            Returns:
                torch.Tensor: Output after the forward pass of the DecoderBlock.
                             Shape: (batch_size, seq_length, d_model).

    """

    def __init__(self, d_model, n_heads, d_k, dropout_prob=0.1):
        super().__init__()

        # Multi-Head Self-Attention layer
        self.mha = MultiHeadAttention(d_model, n_heads, d_k, causal=True)

        # Layer normalization after the first sub-layer
        self.ln1 = nn.LayerNorm(d_model)

        # Layer normalization after the second sub-layer
        self.ln2 = nn.LayerNorm(d_model)

        # Feedforward neural network with GELU activation for the second sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )

        # Dropout layer with specified probability
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, dec_size, attention_mask=None):
        """
        Performs the forward pass through the DecoderBlock.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).
            dec_size (int): Size of the decoder sequence (number of tokens to attend to).
            attention_mask (torch.Tensor, optional): Mask to avoid attending to certain positions.
                                                      Default is None.

        Returns:
            torch.Tensor: Output after the forward pass of the DecoderBlock.
                         Shape: (batch_size, seq_length, d_model).
        """
        # Self-Attention and Layer Normalization
        x = self.ln1(x + self.mha(x, dec_size, attention_mask))

        # Feedforward and Layer Normalization
        x = self.ln2(x + self.ffn(x))

        # Apply dropout
        return self.dropout(x)

class Decoder(nn.Module):
    """
    Decoder is a module representing the decoder of a Transformer architecture.

    Args:
        d_model (int): The dimension of the input features.
        n_heads (int): The number of attention heads in each DecoderBlock.
        d_k (int): The dimension of keys and queries in each attention head.
        n_layers (int): The number of DecoderBlocks in the decoder.

    Attributes:
        transformer_blocks (nn.Sequential): Sequence of DecoderBlocks in the decoder.

    Methods:
        forward(x, dec_size, attention_mask=None):
            Performs the forward pass through the Decoder.

            Args:
                x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).
                dec_size (int): Size of the decoder sequence (number of tokens to attend to).
                attention_mask (torch.Tensor, optional): Mask to avoid attending to certain positions.
                                                          Default is None.

            Returns:
                torch.Tensor: Output after the forward pass of the Decoder.
                             Shape: (batch_size, seq_length, d_model).

    """

    def __init__(self, d_model, n_heads, d_k, n_layers):
        super().__init__()

        # Sequence of DecoderBlocks in the decoder
        self.transformer_blocks = nn.Sequential(*[DecoderBlock(d_model, n_heads, d_k) for _ in range(n_layers)])

    def forward(self, x, dec_size, attention_mask=None):
        """
        Performs the forward pass through the Decoder.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_length, d_model).
            dec_size (int): Size of the decoder sequence (number of tokens to attend to).
            attention_mask (torch.Tensor, optional): Mask to avoid attending to certain positions.
                                                      Default is None.

        Returns:
            torch.Tensor: Output after the forward pass of the Decoder.
                         Shape: (batch_size, seq_length, d_model).
        """
        # Iterate through DecoderBlocks
        for block in self.transformer_blocks:
            x = block(x, dec_size, attention_mask)

        return x  # N, T, hidden_size

class MMT(nn.Module):
    """
    MMT (Modality Modality Transformer) is a module that combines object, OCR, and decoder embeddings
    and processes them through a Transformer-based decoder.

    Args:
        d_model (int): The dimension of the input features.
        n_heads (int): The number of attention heads in each DecoderBlock.
        d_k (int): The dimension of keys and queries in each attention head.
        n_layers (int): The number of DecoderBlocks in the decoder.

    Attributes:
        prev_pred_embeddings (PrevPredEmbeddings): Module for creating embeddings for answer and OCR tokens with positional encoding.
        encoder (Decoder): Transformer-based decoder composed of multiple DecoderBlocks.

    Methods:
        forward(obj_emb, fixed_ans_emb, ocr_emb, prev_inds, attention_mask):
            Performs the forward pass through the MMT.

            Args:
                obj_emb (torch.Tensor): Object embeddings.
                fixed_ans_emb (torch.Tensor): Fixed answer embeddings.
                ocr_emb (torch.Tensor): OCR embeddings.
                prev_inds (torch.Tensor): Indices of previously predicted tokens.
                attention_mask (torch.Tensor): Mask to avoid attending to certain positions.

            Returns:
                torch.Tensor: Output embeddings for OCR tokens and decoder tokens.
                             Shapes: (batch_size, ocr_max_num, d_model), (batch_size, dec_max_num, d_model).

    """

    def __init__(self, d_model, n_heads, d_k, n_layers):
        super().__init__()

        # Module for creating embeddings for answer and OCR tokens with positional encoding
        self.prev_pred_embeddings = PrevPredEmbeddings(d_model)

        # Transformer-based decoder composed of multiple DecoderBlocks
        self.encoder = Decoder(d_model, n_heads, d_k, n_layers)

    def forward(self, obj_emb, fixed_ans_emb, ocr_emb, prev_inds, attention_mask):
        """
        Performs the forward pass through the MMT.

        Args:
            obj_emb (torch.Tensor): Object embeddings.
            fixed_ans_emb (torch.Tensor): Fixed answer embeddings.
            ocr_emb (torch.Tensor): OCR embeddings.
            prev_inds (torch.Tensor): Indices of previously predicted tokens.
            attention_mask (torch.Tensor): Mask to avoid attending to certain positions.

        Returns:
            torch.Tensor: Output embeddings for OCR tokens and decoder tokens.
                         Shapes: (batch_size, ocr_max_num, d_model), (batch_size, dec_max_num, d_model).
        """
        # Create embeddings for answer and OCR tokens with positional encoding
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # Concatenate object, OCR, and decoder embeddings
        encoder_inputs = torch.cat([obj_emb, ocr_emb, dec_emb], dim=1) # batch_size, T (obj + ocr + dec), 768

        # Get the size for each because we will need that in the pointer network
        obj_max_num = obj_emb.size(1)
        ocr_max_num = ocr_emb.size(1)
        dec_max_num = dec_emb.size(1)

        # Offsets of each modality in the joint embedding space
        ocr_begin = obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # Process through the Transformer-based decoder
        encoder_outputs = self.encoder(encoder_inputs, dec_max_num, attention_mask) # N, T, hidden_size

        # mmt_dec_output = encoder_outputs[:, ocr_end:, :] # batch_size, dec_max_num, hidden_size
        # mmt_ocr_output = encoder_outputs[:, ocr_begin: ocr_end, :] # batch_size, ocr_max_num, hidden_size

        return encoder_outputs[:, ocr_begin: ocr_end, :], encoder_outputs[:, ocr_end:, :]

class OcrPtrNet(nn.Module):
    """
    OcrPtrNet is a module for computing attention scores between decoder tokens and OCR tokens
    using linear transformations of input embeddings.

    Args:
        hidden_size (int): The dimension of the input embeddings.

    Attributes:
        hidden_size (int): The dimension of the input embeddings.
        query (nn.Linear): Linear layer for transforming decoder token embeddings.
        key (nn.Linear): Linear layer for transforming OCR token embeddings.

    Methods:
        forward(query_inputs, key_inputs, attention_mask):
            Computes attention scores between decoder tokens and OCR tokens.

            Args:
                query_inputs (torch.Tensor): Decoder token embeddings.
                                            Shape: (batch_size, dec_max_num, hidden_size).
                key_inputs (torch.Tensor): OCR token embeddings.
                                          Shape: (batch_size, ocr_max_num, hidden_size).
                attention_mask (torch.Tensor): Mask to avoid attending to certain OCR positions.
                                              Shape: (batch_size, ocr_max_num).

            Returns:
                torch.Tensor: Attention scores between decoder tokens and OCR tokens.
                             Shape: (batch_size, dec_max_num, ocr_max_num).

    """

    def __init__(self, hidden_size):
        super().__init__()

        # The dimension of the input embeddings
        self.hidden_size = hidden_size

        # Linear layer for transforming decoder token embeddings
        self.query = nn.Linear(hidden_size, hidden_size)

        # Linear layer for transforming OCR token embeddings
        self.key = nn.Linear(hidden_size, hidden_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        """
        Computes attention scores between decoder tokens and OCR tokens.

        Args:
            query_inputs (torch.Tensor): Decoder token embeddings.
                                        Shape: (batch_size, dec_max_num, hidden_size).
            key_inputs (torch.Tensor): OCR token embeddings.
                                      Shape: (batch_size, ocr_max_num, hidden_size).
            attention_mask (torch.Tensor): Mask to avoid attending to certain OCR positions.
                                          Shape: (batch_size, ocr_max_num).

        Returns:
            torch.Tensor: Attention scores between decoder tokens and OCR tokens.
                         Shape: (batch_size, dec_max_num, ocr_max_num).
        """

        # Linear transformations of input embeddings
        scores = self.query(query_inputs) @ self.key(key_inputs).mT
        scores = scores / math.sqrt(self.hidden_size) # batch_size, dec_max_num, ocr_max_num

        # Mask attention scores for certain OCR positions
        scores = scores.masked_fill(attention_mask[:, None, :] == 0, float('-inf'))

        return scores

class M4C(nn.Module):
    """
    M4C (Matters for Captioning) is a multimodal transformer-based model for image captioning.

    Args:
        obj_in_dim (int): The input dimension for object features.
        ocr_in_dim (int): The input dimension for OCR features.
        hidden_size (int): The hidden size of the model.
        n_heads (int): The number of attention heads in the Transformer.
        d_k (int): The dimension of keys and queries in each attention head.
        n_layers (int): The number of layers in the Transformer.
        vocab_size (int): The size of the vocabulary for output captions.
        fixed_ans_emb (torch.Tensor): Fixed answer embeddings.

    Attributes:
        obj_encoder (ObjectEncoder): Module for encoding object features.
        ocr_encoder (OCREncoder): Module for encoding OCR features.
        mmt (MMT): Module for combining object, OCR, and decoder embeddings.
        ocr_ptn (OcrPtrNet): Module for computing attention scores between decoder tokens and OCR tokens.
        classifier (nn.Linear): Linear layer for predicting output captions.
        fixed_ans_emb (torch.Tensor): Fixed answer embeddings.
        finetune_modules (list): List of modules with different learning rates during finetuning.

    Methods:
        get_optimizer_parameters(base_lr):
            Get the optimizer parameters with different/scaled learning rates for finetuning.

        forward(sample, device='cpu'):
            Forward pass through the M4C model.

            Args:
                sample (dict): Input sample containing various modalities and labels.
                device (str): Device to which the model and input should be moved. Default is 'cpu'.

            Returns:
                torch.Tensor: Output scores for predicting captions.
    """

    def __init__(self,
                 obj_in_dim,
                 ocr_in_dim,
                 hidden_size,
                 n_heads,
                 d_k,
                 n_layers,
                 vocab_size,
                 fixed_ans_emb):
        super().__init__()

        # Module for encoding object features
        self.obj_encoder = ObjectEncoder(obj_in_dim=obj_in_dim, hidden_size=hidden_size)

        # Module for encoding OCR features
        self.ocr_encoder = OCREncoder(ocr_in_dim=ocr_in_dim, hidden_size=hidden_size)

        # Module for combining object, OCR, and decoder embeddings
        self.mmt = MMT(d_model=hidden_size, n_heads=n_heads, d_k=d_k, n_layers=n_layers)

        # Module for computing attention scores between decoder tokens and OCR tokens
        self.ocr_ptn = OcrPtrNet(hidden_size=hidden_size)

        # Linear layer for predicting output captions
        self.classifier = nn.Linear(hidden_size, vocab_size)

        # Fixed answer embeddings
        self.fixed_ans_emb = fixed_ans_emb

        # Modules with different learning rates during finetuning
        self.finetune_modules = [{"module": self.obj_encoder.linear_obj_feat_to_mmt_in, "lr_scale": 0.1},
                                 {"module": self.ocr_encoder.linear_ocr_feat_to_mmt_in, "lr_scale": 0.1},
                                 {"module": self.mmt, "lr_scale": 1}]

    def get_optimizer_parameters(self, base_lr):
        """
        Get the optimizer parameters with different/scaled learning rates for finetuning.

        Args:
            base_lr (float): The base learning rate.

        Returns:
            list: List of dictionaries containing parameter groups with different learning rates.
        """
        optimizer_param_groups = []

        # Collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))

        # Remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]

        # Put the default lr parameters at the beginning so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    def forward(self, sample, device='cpu'):
        """
        Forward pass through the M4C model.

        Args:
            sample (dict): Input sample containing various modalities and labels.
            device (str): Device to which the model and input should be moved. Default is 'cpu'.

        Returns:
            torch.Tensor: Output scores for predicting captions.
        """
        # Encode object features
        obj_emb = self.obj_encoder(sample['obj_boxes'].to(device), sample['obj_features'].to(device))

        # Encode OCR features
        ocr_emb = self.ocr_encoder(sample['ocr_boxes'].to(device),
                               sample['ocr_token_embeddings'].to(device),
                               sample['ocr_rec_features'].to(device),
                               sample['ocr_det_features'].to(device))

        # Create decoder inputs
        dec_input = sample['labels'].clone().detach().roll(shifts=1, dims=1)
        dec_input[:, 0] = 0 # <s> token

        # Combine object, OCR, and decoder embeddings
        mmt_ocr_output, mmt_dec_output = self.mmt(obj_emb,
                                              self.fixed_ans_emb,
                                              ocr_emb,
                                              dec_input.to(device),
                                              sample['join_attn_mask'].to(device))

        ocr_begin = obj_emb.size(1)
        ocr_end = obj_emb.size(1) + mmt_ocr_output.size(1)

        # Cat the prediction from mmt and pointer network
        scores = torch.cat([self.classifier(mmt_dec_output),
                        self.ocr_ptn(mmt_dec_output, mmt_ocr_output, sample['join_attn_mask'][:, ocr_begin: ocr_end].to(device))],
                        dim=-1)

        return scores

def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    results = F.embedding(batch_offsets + inds, x.view(batch_size * length, dim)) # batch_size, T, hidden_size
    return results