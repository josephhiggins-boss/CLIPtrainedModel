# What You Need to Install



You only need 5 packages to run the basic scoring:

'''
pip install torch transformers pillow numpy scipy
'''



## What Each Package Does

- **torch** - This is PyTorch, the AI framework. It's what runs the neural network
- **transformers** - This gives us the CLIP model that understands images
- **pillow** - This opens and reads image files (JPG, PNG, etc.)
- **numpy** - Does math on the image pixels
- **scipy** - Makes some image processing faster


## What Files You Need

### Required

1. **Your trained model file** (`.pth` file)
   - This is the AI model we trained
   - 587mb
   - Put it somewhere easy to find
