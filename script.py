#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----------------------
# |Continued from Laptop|
# -----------------------

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)

# The model is now trained and therefore it can be used to make predictions about images

predictions = model.predict(test_images)

# Plot the first 25 images with the predicted label and the actual label. 
# Correct predictions will be coloured green while incorrect coloured red.
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	predicted_label = np.argmax(predictions[i])
	true_label = test_labels[i]
	if predicted_label[i] == test_labels[i]:
		color = 'green'
	else:
		color = 'red'
	plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[true_label]), color=color)

img = test_images[0]
img = (np.expand_dims(img,0))
predictions = model.predict(img)

