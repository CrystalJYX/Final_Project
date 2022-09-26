# Final_Project: User Satisfaction Estimation in Task-oriented Dialogue Systems

#### Document description
MWOZ and SGD files respectivaly contain 6 models, which are USDA (SE_DAR)[1], SE_DAR_DSR and SE_DAR_SDE for datasets (SGD and MWOZ)[2] based on three-level scale and five-class scale. data processing file contains plots used in this report.
* USDA (SE_DAR)[1]: two-task-learning model for User Satisfaction Estimation (SE) and User Dialogue Act Recognition (DAR)  
* SE_DAR_DSR: three-task-learning model for User Satisfaction Estimation (SE), User Dialogue Act Recognition (DAR) and User Dialogue Satisfaction Recognition (DSR)  
* SE_DAR_SDE: three-task-learning model for User Satisfaction Estimation (SE), User Dialogue Act Recognition (DAR) and User Satisfaction Difference Estimation (SDE)  

In each file in MWOZ and SGD files, the file's name number_model represents the model for datasets with different scales.  
For example, 3_SE_DAR_SDE in MWOZ file represents the code and results for SE_DAR_SDE model for MWOZ data based on the three-level scale.

Each processed dataset has been compressed and placed in individual files (Final_Project/MWOZ/number_model/dataset/ ), please uncompress it and do not move the dataset before running models

#### Terminal commands
These codes need one GPU to run around 10min for each epoch. You can use the following command to run the model:  
!pip install torch  
!pip install transformers  
!pip install pytorch-crf   
import torch   
cd /content/3_SE_DAR_DSR/mtl 
##### you should change the file name '3_SE_DAR_DSR' for other models   
!CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=8 --data=mwoz --model=SE_DAR_DSR  
##### you should change the dataset name 'mwoz' and the model name 'SE_DAR_DSR' for other models 

#### References
[1]: [User Satisfaction Estimation with Sequential Dialogue Act Modeling in Goal-oriented Conversational Systems](https://arxiv.org/pdf/2202.02912.pdf)    
[code](https://github.com/dengyang17/USDA)  
[2]: [Simulating user satisfaction for the evaluation of task-oriented dialogue systems](https://arxiv.org/pdf/2105.03748.pdf)   
[dataset](https://github.com/sunnweiwei/user-satisfaction-simulation)   
This project is an optimisation of the above two reports and code.  

#### The results for datasets with three-level scale are shown below:
<img width="559" alt="截屏2022-09-26 11 01 55" src="https://user-images.githubusercontent.com/69834165/192185649-bae00cdd-d94e-4c57-b486-82a37c6e2e30.png">
<img width="549" alt="截屏2022-09-26 11 03 21" src="https://user-images.githubusercontent.com/69834165/192185797-60f07e78-31f8-429f-b044-f5cdca6d6944.png">

#### The results for datasets with five-class scale are shown below:  
<img width="568" alt="截屏2022-09-26 11 04 27" src="https://user-images.githubusercontent.com/69834165/192186011-b1c7c639-769b-4fb3-ae7f-480cf650b1bd.png">
<img width="557" alt="截屏2022-09-26 11 04 37" src="https://user-images.githubusercontent.com/69834165/192186018-a59a37e9-1ad1-4325-83fc-7fb8680d7c71.png">
<img width="575" alt="截屏2022-09-26 11 04 53" src="https://user-images.githubusercontent.com/69834165/192186020-b882c58a-b0df-4ddc-903f-78e4bb046990.png">
