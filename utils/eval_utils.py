import math
import torch


def evaluate_rand(model, image, scale, num_samples_test):
    # evaluate images (randomized smoothing)
    num_images = image.shape[0]
    preds = None
      
    for i in range(num_images):
        batch_size = 50
        num_batches = int(math.ceil(num_samples_test / batch_size))
        ps = None

        for j in range(num_batches):
            bstart = j * batch_size
            bend = min(bstart + batch_size, num_samples_test)
            
            image_noise = image[i].unsqueeze(0).repeat(bend - bstart, 1, 1, 1)
            image_noise = image_noise + scale * torch.randn_like(image_noise)
            
            output = model(image_noise)
            p = torch.max(output, 1)[1]
            ps = p if ps is None else torch.cat([ps, p], dim=0)

        count = torch.bincount(ps, minlength=output.shape[1])
         
        pred = torch.argmax(count)
        preds = pred.view(1) if preds is None else torch.cat([preds, pred.view(1)], dim=0)
       
    return preds

