# GCN-SOAN: Improving Peer Assessment with Graph Convolutional Networks
<section>
  <h2>Table of Contents:</h2>
  <ul>
  <li><a href="#introduction"><b>Introduction</b></a></li>
  <li><a href="#baselines"><b>Baselines</b></a></li>
  <li><a href="#run"><b>How to Run Experiments</b></a></li>
  </ul>
</section>

<!-- introduction -->
<section>
  <h2 id="introduction">Introduction</h2>
  <p>
  <b>GCN-SOAN</b> is a semi-supervised graph convolutional network, leveraging the power of GCNs and the flexibility of multi-relational weighted networks, coupled with experts' evaluations as ground truths.
  <br>
</section>

<!-- baselines -->
<section>
  <h2 id="baselines">Baselines</h2>
  <ol>
    <li><a href="https://en.wikipedia.org/wiki/Average">Average:</a> Take the average of the peer assessments to estimate ground-truth valuations.</li>
    <li><a href="https://en.wikipedia.org/wiki/Median">Median:</a> Take the median of the peer assessments to estimate ground-truth valuations.</li>
    <li><a href="https://arxiv.org/pdf/1405.7192.pdf">PeerRank:</a> The PeerRank model is an iterative model motivated by the Google PageRank algorithm. In PeerRank, the grade of a user relies on the grade of its graders, and the grade of the graders also relies on the grade of other graders.</li>
    <li><a href="https://arxiv.org/pdf/1307.2579.pdf">TunedModel:</a> In TunedModels, the authors formulate peer assessment as a probabilistic model by defining users' true score, graders' bias, and graders' reliability with peer grades as the only inputs of the model. They framed the probabilistic model as a posterior distribution problem over the latent variables given on observed peer grades.</li>
    <li><a href="https://ieeexplore.ieee.org/abstract/document/8252331">RankwithTA:</a> RankwithTA is an iterative algorithm that incorporates experts' grades as ground truth for a portion of the submissions. In RankwithTA, the final grade of each user depends on both their performance in answering their own submission and the quality of their peer assessments.</li>
    <li><a href="https://arxiv.org/pdf/1308.5273.pdf">Vancouver:</a> Vancouver is a crowdsourcing algorithm that aggregates peer grades leveraging a reputation system that captures users' grading accuracy. The Vancouver algorithm is motivated by expectation maximization (EM) techniques and uses the variance minimization principle to estimate students' performances in their grading ability to control the influence each grader exerts on the final consensus grade of each submission.</li>
  </ol>
</section>

<!-- How to Run Experiments -->
<section>
  <h2 id="run">How to Run Experiments?</h2>
  <h3>Required Packages</h3>
  <p><strong>The easy/recommend way:</strong> We recommend to use a virtualenv to avoid conflicting with your previously installed packages. To create a virtualenv and install all required packages use the following commands:
    <ol>
      <li>Navigate to the repository folder using <code>cd GCN-SOAN</code></li>
      <!-- <li>pip3 install virtualenv</li> -->
      <li>Create a new virtual environment for python: <code>virtualenv gcn_soan_env</code></li>
      <li>Activate the created virtual environment: <code>source gcn_soan_env/bin/activate</code></li>
      <li>Install required packages: <code>./setup.sh</code> in Linux OS, and <code>./setup_windows.sh</code> in Windows OS </li>
    </ol>
  </p>
  <p>* Make sure to have execute permission for setup.sh file: chmod +x setup.sh</p>
  <p>* If you get any error during installing Pytorch on your system, the Pytorch community is the best place to look for: <a href="https://discuss.pytorch.org/">Pytorch Community</a></p>
  <p><strong>The hard way:</strong> Install the following packages separately using pip</p>

    matplotlib==3.3.4
    networkx==2.5.1
    numpy==1.19.5
    pandas==1.1.5
    scikit-learn==0.24.2
    scipy==1.5.4
    torch==1.8.2+cu102
    torch-cluster==1.5.9
    torch-geometric==2.0.1
    torch-scatter==2.0.9
    torch-sparse==0.6.12
    torch-spline-conv==1.2.1
    torchaudio==0.8.2
    torchvision==0.9.2+cu102
  <h3>Run all studies at once</h3>
  <p>
    <ol>
        <li>Give execute permission for run_studies.sh file: <code>chmod +x run_studies.sh</code></li>
        <li>Run the bash script: <code>run_studies.sh</code></li>
    </ol>
  </p>
  <h3>Run simulated studies in paper</h3>
  <p>
    There are 6 studies conducted in the paper, each investigating the impact of one factor on the results. To run each of them separately:
    <ol>
      <li>First you need to update the set_study.py file with a unique identifier defined for each study.</li>
      <li>Then you should run: <code>python3 main.py</code></li>
    </ol>
    For example, to run the first study which examines the impact of number of graders per item, you should update the set_study.py file with: <code>study = 'number_of_graders'</code> and then run: <code>python3 main.py</code><br><br>
  </p>
  <h3>Run a customized simulated study</h3>
  <p>
    <ol>
      <li>First you need to update the set_study.py file with a unique identifier called "custom".</li>
      <li>Then, choose your desired parameters in set_study.py for "custom_study" variable.</li>
      <li>Finally, you should run: <code>python3 main.py</code></li>
    </ol>
  </p>
  <p>
    The above command generates corresponding synthetic datasets, run GCN-SOAN and all baselines, and save the results in data/studies directory. The plots also will be generated and saved in pdf file in the corressponding directory.
  </p>

  <h3>Run experiments on real data</h3>
  <ol>
    <li>The dataset is not publicly available, so you should request it from the following link: <a href="http://www.tml.cs.uni-tuebingen.de/team/luxburg/code_and_data/peer_grading_data_request_new.php">Request for real dataset</a></li>
    <li>You need to locate the 4 main files of the dataset containing peer grades, self grades, TA grades, and max grades in <code>data/real_data_raw</code> directory.</li>
    <li>Set the study in set_study to the desired experiment you want, it can be one of the followings: <br> 
      <code>real_data_peer_evaluation</code><br>
      <code>real_data_peer_and_self_evaluation</code>
    </li>
    <li>Run <code>python3 main.py</code></li>
    <li>The results will be saved in the following file: <code>data/real_data_processed/rmse_models.csv</code></li>
  </ol>

  <h3>List of unique identifiers for studies:</h3>
  <p>
    <ul>
      <li>
        <b>Study 1 - Number of graders per item: </b> The unique identifier for this study is <strong>"number_of_graders"</strong>
      </li>
      <li>
        <b>Study 2 - Bias parameter: </b> The unique identifier for this study is <strong>"graders_bias"</strong>
      </li>
      <li>
        <b>Study 3 - Ground truth mean: </b> The unique identifier for this study is <strong>"ground_truth_distribution"</strong>
      </li>
      <li>
        <b>Study 4 - Reliability parameter: </b> The unique identifier for this study is <strong>"working_impact_grading"</strong>
      </li>
      <li>
        <b>Study 5 - Strategic peer grade generation: </b> The unique identifier for this study is <strong>"erdos"</strong>
      </li>
      <li>
        <b>Study 6 - Homophily: </b> The unique identifier for this study is <strong>"homophily"</strong>
      </li>
      <li>
        <b>Study 7 - Real data peer evaluation: </b> The unique identifier for this study is <strong>"real_data_peer_evaluation"</strong>
      </li>
      <li>
        <b>Study 8 - Real data peer and self evaluation: </b> The unique identifier for this study is <strong>"real_data_peer_and_self_evaluation"</strong>
      </li>
      <li>
        <b>Study 9 - Custom: </b> The unique identifier for this study is <strong>"custom"</strong>
      </li>
    </ul>
  </p>
</section>



