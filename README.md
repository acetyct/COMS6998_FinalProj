# COMS6998_FinalProj
This is the repo for Final Project of COMS6998

super final actual is the final version of code for super model


A description of the project  <br>
In this project we set out to optimize convolution neural networks through exploring the techniques of pruning, mixed precision, quantization and synchronous training. 


A description of the repository <br>
This Repo contains the codes, any relevant training log, and all the models that we produced during the project. <br>
The training log and models for ResNet 20 and ResNet 44 are stored in seperate folders respectively.<br>

Under the folder codes, we include all training codes for the trials of exploration, for example, testing constant sparsity, poly-sparsity, and distributed training, using CIFAR-10 and CIFAR-100 datasets. Note that the code for producing the final super model is not included in this folder. <br>

The code for producing the final super model is named super_final_actual.ipynb. <br>

We tried two differnt super model, one with sparsity 0.6 and one with sparsity 0.7. As mentioned in the presentation, the hyperparameters are found through all the many trials we run. We find the sparsities that would give the highest validation accuracy. We tried both 0.6 and 0.7 to see the impact of slightly variaing the sparsity hyperparameter. The training log and trained models are stored under /Super_Model/super_0.6 and  /Super_Model/super_0.7 <br>




Example commands to execute the code   <br>     
We included the codes as jupyter notebook, and all the notebooks can be run on GCP VMs. Note that for the distributed training of the super model, it need to be run on Vertex AI. The set up of Vertex AI is described in detail in this video: https://www.youtube.com/watch?v=rAGauhXYgw4&list=WL&index=1 . When logged into Vertex AI workbench, just hit the "JupyterLab" button to getinto a jupyterlab, and upload the jupyternotebook using the UI. Then, on the upper-right corner, select the machine configuration. We used 4 CPUs, 15 GB RAM, and 1 Tesla V100. Then, the distributed training notebook can be run normally.<br>

Results (including charts/tables) and your observations  <br>
