# Heart Disease Prediction Using Neural Networks

This project applies various optimization techniques to train a neural network for predicting heart disease. The dataset contains attributes related to heart health and a binary target indicating the presence of heart disease.

## Objectives
- Compare the performance of four optimization techniques: Gradient Descent, Genetic Algorithm, Simulated Annealing, and Randomized Hill Climbing.
- Use Particle Swarm Optimization (PSO) as an additional technique to optimize the neural network weights.

## Project Structure
```plaintext
heart-disease-prediction/
├── data/
│   └── heart_statlog_cleveland_hungary_final.csv  # Dataset
├── docs/
│   └── documentation.pdf                         # Project documentation
├── media/
│   ├── genetic_algorithm_loss_curve.png          # Loss curve for Genetic Algorithm
│   ├── gradient_descent_loss_curve.png           # Loss curve for Gradient Descent
│   ├── randomized_hill_climb_loss_curve.png      # Loss curve for Randomized Hill Climbing
│   └── simulated_annealing_loss_curve.png        # Loss curve for Simulated Annealing
├── report/
│   ├── Report1.docx                 # Detailed project report
│   └── Report2.docx                       
├── scripts/
│   └── heart-disease-prediction/
│       ├── gradient_descent.py                  # Gradient Descent implementation
│       ├── genetic_algorithm.py                 # Genetic Algorithm implementation
│       ├── particle_swarm_optimization.py       # PSO implementation
│       ├── randomized_hill_climb.py            # Randomized Hill Climbing implementation
│       ├── simulated_annealing.py              # Simulated Annealing implementation
├── README.md                                     # This file
└── requirements.txt                              # Python dependencies
```

## Files in This Repository
- **Data**:
  - data/heart_statlog_cleveland_hungary_final.csv: Dataset used for training and testing.
- **Documentation**:
  - docs/documentation.pdf: Detailed dataset description.
- **Reports**:
  - report/Report1.docx: Main project report with results and discussions.
  - report/Report2.docx: Bonus report using PSO.
- **Scripts**:
  - scripts/gradient_descent.py: Neural network trained with Gradient Descent.
  - scripts/genetic_algorithm.py: Neural network trained with Genetic Algorithm.
  - scripts/simulated_annealing.py: Neural network trained with Simulated Annealing.
  - scripts/randomized_hill_climb.py: Neural network trained with Randomized Hill Climbing.
  - scripts/particle_swarm_optimization.py: Neural network trained with Particle Swarm Optimization (PSO).
- **Visualizations**:
  - Loss curves and learning curves for all methods are in the media/ folder.

## Results
| Method                 | Accuracy | Sensitivity | Specificity | AUC  |
|------------------------|----------|-------------|-------------|-------|
| Gradient Descent       | 0.84     | 0.88        | 0.79        | 0.83  |
| Genetic Algorithm      | 0.55     | 0.21        | 0.96        | 0.58  |
| Simulated Annealing    | 0.85     | 0.94        | 0.75        | 0.84  |
| Randomized Hill Climb  | 0.84     | 0.89        | 0.79        | 0.84  |
| Particle Swarm Optimization (PSO) | Results may vary due to stochasticity |

## Key Insights
- Simulated Annealing and Gradient Descent performed the best overall.
- Genetic Algorithm struggled with sensitivity and AUC but excelled in specificity.
- Particle Swarm Optimization is promising for weight optimization but requires tuning.

## Tools and Libraries
- **Python Libraries**:
  - numpy, pandas, matplotlib
  - pyswarms for PSO
  - mlrose_hiive for optimization algorithms
  - scikit-learn for model evaluation
- **Tools**:
  - Word for reporting
  - Kaggle for dataset exploration

## Instructions to Run the Code
1. Clone the repository:
   git clone https://github.com/3bdulah/heart-disease-prediction.git
   cd heart-disease-prediction

2. Set up the environment:
   python -m venv .venv 
   ### For Windows:
   .\.venv\Scripts\activate
   ### For macOS/Linux:
   source .venv/bin/activate


3. Install dependencies:
   pip install -r requirements.txt

4. Run the scripts:

   python scripts/heart-disease-prediction/gradient_descent.py

   python scripts/heart-disease-prediction/genetic_algorithm.py

   python scripts/heart-disease-prediction/simulated_annealing.py

   python scripts/heart-disease-prediction/randomized_hill_climb.py

   python scripts/heart-disease-prediction/particle_swarm_optimization.py

## Additional Notes
- All reports and documentation are in the report/ and docs/ folders.
- Media visualizations are stored in the media/ folder.
- Dynamic file paths are implemented using os.path for portability.
