# Experimentation with neural network

The layers used where changed several times trying to increase the performance and reduce the overfitting. First of all I needed to make sure the model run, so I used only one layer with only 8 units. Then I increased the number of units, with some success.

What really made a big difference was the inclusion of the convolution and the pooling. I included this twice and finally used a dropout to reduce overfitting.

I spent some time trying to get better results but I wasn't able to do it. The best I could get was an accurracy in testing of <span style="color:green">**0.9857**</span>. The layers used were:

- Reescaling: 1./255
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Flatten
- Dense, 128 units, activation relu
- Dropout, 0.2
- Dense, 43 units, activation sigmoid. OUTPUT

In chronological order what I tried was:

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

> **Added reescaling :** The pixel values of the images were reescaled to the [0, 1] interval. The performance increased, there is some overfitting.

- Reescaling: 1./255
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Flatten
- Dense, 128 units, activation relu
- Dense, 43 units, activation sigmoid. OUTPUT

<center>

|          | Training | Testing |
| -------- | -------- | ------- |
| Loss     | 0.0027   | 0.0087  |
| Accuracy | 0.9941   | 0.9664  |

</center>

---

> **Added convolution and reescaling a second time :** The performance in the testing increased a little bit, there is some overfitting.

- Reescaling: 1./255
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Flatten
- Dense, 128 units, activation relu
- Dense, 43 units, activation sigmoid. OUTPUT

<center>

|          | Training | Testing |
| -------- | -------- | ------- |
| Loss     | 0.0025   | 0.0069  |
| Accuracy | 0.9939   | 0.9796  |

</center>

---

> **Added dropout to reduce overfitting:** The performance in the testing decreased, but there is no overfitting.

- Reescaling: 1./255
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Flatten
- Dense, 128 units, activation relu
- Dropout, 0.5
- Dense, 43 units, activation sigmoid. OUTPUT

<center>

|          | Training | Testing |
| -------- | -------- | ------- |
| Loss     | 0.0075   | 0.0053  |
| Accuracy | 0.9687   | 0.9820  |

</center>

---

> **Reducing dropout:**

- Reescaling: 1./255
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Convolution: 32 filters, size 3x3
- Max pooling: size 2x2
- Flatten
- Dense, 128 units, activation relu
- Dropout, 0.2
- Dense, 43 units, activation sigmoid. OUTPUT

<center>

|          | Training | Testing |
| -------- | -------- | ------- |
| Loss     | 0.0035   | 0.0044  |
| Accuracy | 0.9889   | 0.9857  |

</center>

---
