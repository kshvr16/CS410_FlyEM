#!/usr/bin/env python
# coding: utf-8

# In[2]:


@staticmethod
def apply_threshold(original_image, threshold_value):
        # returns the threshold image of the original image for the given threshold value
        result_image = original_image.copy()
        result_image[result_image < threshold_value] = 0
        return result_image


# In[4]:


@staticmethod
def apply_region_labelling(original_image):
        # returns the labeled regions and number of regions
        labeled_result, nr_objects_result = mh.label(original_image)
        return labeled_result, nr_objects_result


# In[5]:


def apply_threshold(original_image, threshold_value):
    result_image = original_image.copy()
    result_image = result_image.from_numpy(result_image)
    result_image = result_image.to('cuda')
    result_image.masked_fill_(result_image < threshold_value, 0)
    result_image = result_image.cpu().numpy()
    return result_image


# In[9]:


@staticmethod
def remove_small_regions(labeled, region_size):
 sizes =torch.bincount.labeled_size(labeled)
 too_small= np.where(sizes < region_size)
 labeled_result=labeled.remove_regions(labeled, too_small)
 return labeled_result


# In[14]:


@staticmethod
def get_binary_mask(labeled):
 result_mask=(labeled_only_big > 0).copy()
 result_mask[result_mask > 0]= binary_mask.max().item()


# In[ ]:




