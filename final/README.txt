I think the model itself is done. But I haven't touched any of the training code.
The model may have issues with typecasts. We may have to make everything float16 as cast isn't differentiable.
The model may also have issues with dimentionality but I think (hope) that's unlikely.
To-do:
	* Find dataset(s)
	* Partition dataset(s) into smaller groups (we could use multiple different data-sets but I think this will be easier and ok) leave at
		least one control group that we won't train on at all until we're "done" with the conv-net
	* Write double for loop. Outer for loop should be a regular training loop. Inner for loop should loop through the different training
		data sets and final dense_layers associated with each dataset.
	* Before we do any seriosu training, we need to make sure saving and loading models works as expected.