import csv
import os

import torch
import imageio

import kernel3a2ac1615a as dataset
import kernel447322982f as wheat_detector

def main():
    wheat_net = wheat_detector.WheatDetector()
    wheat_net.load_state_dict(
        torch.load(
            "../input/bestmodel4/wheat_detector_model_46185.pt",
            map_location=torch.device('cpu')))
    mean = torch.tensor([84.4350, 76.9273])
    std = torch.tensor([35.5535, 33.8532])

    test_dir = '/kaggle/input/global-wheat-detection/test'
    test_images = os.listdir(test_dir)
    with open('/kaggle/working/submission.csv', 'w', newline='') as submission:
        submission_writer = csv.writer(submission, delimiter=',')
        submission_writer.writerow(["image_id", "PredictionString"])
        for img_name in test_images:
            image_id = img_name.split('.')[0]
            img = torch.tensor(
                imageio.imread(os.path.join(f"{test_dir}",
                                            f"{img_name}"))).permute(2, 0, 1)
            img = img.float() / 255
            img = img.unsqueeze(0)
            padded_image = torch.zeros((1, 3, 1024, 1024))
            _, _, img_height, img_width = img.shape
            padded_image[:, :, 0:min(1024, img_height),
                         0:min(1024, img_width)] = img[:, :,
                                                       0:min(1024, img_height),
                                                       0:min(1024, img_width)]

            with torch.no_grad():
                scores, bboxes = wheat_net(padded_image)
                
            decoded_bboxes = dataset.decode_bboxes(bboxes, mean, std)
            decoded_bboxes = torch.clamp(decoded_bboxes, 0, 1023)
            
            selected = dataset.select_bboxes(decoded_bboxes, scores, 65, 65, 0.4)
            #selected = dataset.select_bboxes_nms(decoded_bboxes, scores, 0.4)

            submission_row = ""
            for y in range(decoded_bboxes.shape[1]):
                for x in range(decoded_bboxes.shape[2]):
                    #if (y * decoded_bboxes.shape[1] + x) in selected:
                    if selected[0, 0, y, x]:
                        y_min, x_min, height, width = decoded_bboxes[0, y, x]
                        submission_row = submission_row + f"{scores[0, 0, y, x].item():.2} {int(x_min.item())} {int(y_min.item())} {int(width.item())} {int(height.item())} "
            submission_writer.writerow([image_id, submission_row])


if __name__ == "__main__":
    main()
