# Reduce the learning rate for more gradual fine-tuning
for param_group in optimizer.param_groups:
    param_group["lr"] *= 0.5

print("Updated learning rates:")
for i, param_group in enumerate(optimizer.param_groups):
    print(f"Group {i + 1}: {param_group['lr']}")

# Train for one more epoch
model.train()

train_loss = 0.0
correct = 0
total = 0

for images, labels in tqdm(train_loader, desc="Fine-tuning Epoch 3"):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    train_loss += loss.item()

    predictions = outputs.argmax(dim=1)
    total += labels.size(0)
    correct += (predictions == labels).sum().item()

train_accuracy = 100 * correct / total
train_loss = train_loss / len(train_loader)

print("\nEpoch 3 Training Results")
print(f"Training Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_accuracy:.2f}%")
