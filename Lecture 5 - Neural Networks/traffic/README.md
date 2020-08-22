# Experimentation with neural network

The layers used where changed several times trying to increase the performance and reduce the overfitting. In chronological order what was tried was:

---

> **"Get it to work"**: the first step that I took was to make sure I got it to work. I had several problems with the libraries that tensorflow required. Apart from that, I had forgotten to add the flatten layer. This first NN perfomed really badly:

<center>

|          | Training | Testing |
| -------- | -------- | ------- |
| Loss     | 0.1207   | 0.1177  |
| Accuracy | 0.0572   | 0.0550  |

</center>

Layers used:

- Flatten.
- Dense, 8 units, activation relu
- Dense, 43 units, activation sigmoid. OUTPUT

---

> **_Increased number of units in the hidden layer_** Number of units increased from 8 to 128. The performance increased a little.

<center>

|          | Training | Testing |
| -------- | -------- | ------- |
| Loss     | 0.0747   | 0.0734  |
| Accuracy | 0.4080   | 0.4316  |

</center>

- Flatten
- Dense, 128 units, activation relu
- Dense, 43 units, activation sigmoid. OUTPUT

---

> **Added convolution and max pooling:** The performance increased a lot, but there is some overfitting.

- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Flatten
- Dense, 128 units, activation relu
- Dense, 43 units, activation sigmoid. OUTPUT

<center>

|          | Training | Testing |
| -------- | -------- | ------- |
| Loss     | 0.0168   | 0.0347  |
| Accuracy | 0.9613   | 0.9304  |

</center>

---
