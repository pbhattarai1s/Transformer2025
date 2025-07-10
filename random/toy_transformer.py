import torch
import torch.nn as nn
import matplotlib.pyplot as plt

### define inputs 
batch_size = 16
num_particles = 10
num_features = 3
embed_dim = 64

## step 1: create a random input tensor
input_tensor = torch.randn(batch_size,num_particles, num_features)  # (batch_size, num_particles, num_features) 

print("Input Tensor Shape:", input_tensor.shape)
print("Input Tensor:", input_tensor)

### step 2: embedding into a higher dimension
embedding_layer = nn.Linear(num_features, embed_dim)
input_tensor_embedded= embedding_layer(input_tensor)  #changes the featire i to i*W+b where W is a weight matrix and b is a bias vector, this is a linear transformation and are randomly initialized
print("Embedded Tensor Shape:", input_tensor_embedded.shape)
print("Embedded Tensor:", input_tensor_embedded)

### step 3: position encoding, i am assuming that my particles are ordered in pt and i want to add positional information for that
# doing it the way described in the paper, i.e. sin and cos functions
def positional_encoding(particle_len, embedded_dim):
    pos = torch.arange(particle_len).unsqueeze(1)  # shape (particle_len, 1)

    i = torch.arange(embedded_dim).unsqueeze(0)  # shape (1, embedded_dim)

    # let's compute the denominators : 1000^(2i/d_model)**
    denom = torch.pow(10000, (2 * (i // 2)) / embedded_dim)  # shape (1, embedded_dim)

    angles = pos / denom  # shape (particle_len, embedded_dim)
    #applying sin to even indices and cos to odd indices
    angles[:, 0::2] = torch.sin(angles[:, 0::2])
    angles[:, 1::2] = torch.cos(angles[:, 1::2])
    print("angles shape: ", angles.shape)
    
    return angles 

# let's apply the positional encoding to the embedded tensor
pos_encoded = positional_encoding(num_particles, embed_dim)
print("Positional Encoding Shape:", pos_encoded.shape)
print("Positional Encoding:", pos_encoded)  

reshaped_pos_encoded = pos_encoded.unsqueeze(0)
print ("Reshaped Positional Encoding Shape:", reshaped_pos_encoded.shape)
print("Reshaped Positional Encoding:", reshaped_pos_encoded)

# adding the positional encoding to the embedded tensor
input_tensor_embedded += reshaped_pos_encoded  # unsqueeze to match batch size
print("Positional Encoded Embedded Tensor Shape:", input_tensor_embedded.shape)
print("Positional Encoded Embedded Tensor:", input_tensor_embedded)

### step 4: creating a transformer encoder layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=8,
    dim_feedforward=256,
    batch_first=True
)
transformer_encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=6
)
encoder_output = transformer_encoder(input_tensor_embedded)
print("Transformer Encoder Output Shape:", encoder_output.shape)
print("Transformer Encoder Output:", encoder_output)

""" --------- *************** Manual implementation of a transformer encoder layer for my understanding *************** -------------###
        # ### step 4a: let's create a multi-head attention layer
        # multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        # attn_output, attn_weights = multihead_attn.forward(input_tensor_embedded, input_tensor_embedded, input_tensor_embedded)
        # print("Attention Output Shape:", attn_output.shape)
        # print("Attention Output:", attn_output)
        # print("Attention Weights:", attn_weights)
        # print("Attention Weights Shape:", attn_weights.shape)

        # ### step 4b: let's add the residual connection and layer normalization
        # residual_output = input_tensor_embedded + attn_output
        # print("Residual Output Shape:", residual_output.shape)
        # print("Residual Output:", residual_output)
        # layer_norm = nn.LayerNorm(64)
        # normalized_output = layer_norm(residual_output)
        # print("Normalized Output Shape:", normalized_output.shape)
        # print("Normalized Output:", normalized_output)

        # ## step 4c: let's create a feedforward layer
        # feedforward = nn.Sequential(
        #     nn.Linear(64, 256),  # first linear layer
        #     nn.ReLU(),           # activation function
        #     nn.Linear(256, 64)   # second linear layer
        # )
        # ff_output = feedforward(normalized_output)
        # print("Feedforward Output Shape:", ff_output.shape)
        # print("Feedforward Output:", ff_output) 
        # # step 4d: adding the residual connection and layer normalization again
        # residual_ff_output = normalized_output + ff_output
        # print("Residual Feedforward Output Shape:", residual_ff_output.shape)
        # print("Residual Feedforward Output:", residual_ff_output)
        # normalized_ff_output = layer_norm(residual_ff_output)
        # print("Normalized Feedforward Output Shape:", normalized_ff_output.shape)
        # print("Normalized Feedforward Output:", normalized_ff_output)
    ### --------- *************** Manual implementation of a transformer encoder layer for my understanding *************** -------------###
"""
### step 5: let's pool the output of the transformer encoder
pooled_output = encoder_output.mean(dim=1)  # pooling over the sequence length, this gives us a single vector for each batch 
print("Pooled Output Shape:", pooled_output.shape)
print("Pooled Output:", pooled_output)

### step 6: let's create a linear layer to map the pooled output to a desired output dimension
classifier = nn.Linear(64, 1)  # we only want a single output for classification # 10 is the number of particles, 64 is the embedding dimension 
logits = classifier(pooled_output)  # this gives us the logits for the classification task
print("Logits Shape:", logits.shape)
print("Logits:", logits)

### step 7: let's apply a sigmoid activation to get the probabilities
sigmoid = nn.Sigmoid()
probabilities = sigmoid(logits)  # this gives us the probabilities for the classification task
print("Probabilities Shape:", probabilities.shape)
print("Probabilities:", probabilities)

### step 8: let's create a fake target 
# binary labels for each event
target = torch.randint(0, 2, (batch_size,1)).float()

### step 9: let's compute the loss
criterion = nn.BCEWithLogitsLoss()  # more stable than sigmoid + BCELoss
optimizer = torch.optim.Adam(
    list(embedding_layer.parameters()) +
    list(transformer_encoder.parameters()) +
    list(classifier.parameters()),
    lr=0.001
)

### step 10: let's create a toy validation set 
val_input_tensor = torch.randn(batch_size, num_particles, num_features)
val_target = torch.randint(0, 2, (batch_size, 1)).float()

### ---------- TRAINING LOOP ----------
losses = []
val_losses = []
num_epochs = 20
print("Starting Training Loop...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # forward pass
    input_tensor_embedded = embedding_layer(input_tensor)
    input_tensor_embedded += reshaped_pos_encoded  # broadcasted to (16,10,64)
    encoder_output = transformer_encoder(input_tensor_embedded)
    pooled_output = encoder_output.mean(dim=1)  # (16,64)
    logits = classifier(pooled_output)          # (16,1)
    
    # compute loss
    loss = criterion(logits, target)
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    losses.append(loss.item())
    
    # backward + update
    loss.backward()
    optimizer.step()

    # validation step
    with torch.no_grad():
        val_input_tensor_embedded = embedding_layer(val_input_tensor)
        val_input_tensor_embedded += reshaped_pos_encoded  # broadcasted to (16,10,64)
        val_encoder_output = transformer_encoder(val_input_tensor_embedded)
        val_pooled_output = val_encoder_output.mean(dim=1)  # (16,64)
        val_logits = classifier(val_pooled_output)          # (16,1)
        
        val_loss = criterion(val_logits, val_target)
        val_losses.append(val_loss.item())
        print(f"Validation Loss: {val_loss.item():.4f}")

# ---------- TRAINING LOOP END ----------
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), losses, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Validation Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.savefig("training_loss.png")
plt.show()