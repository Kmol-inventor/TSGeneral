from tsgeneral import Inspector, Pipeline, StatefulPipeline
from test.test_data.generators import generate_eeg_trials_by_samples

data = generate_eeg_trials_by_samples(n_trials=10, n_samples=500, seed=42)

pipeline = Pipeline()
spip = StatefulPipeline()


pipeline.add_stage("Raw")
spip.add_stage("Raw")






inspector = Inspector(data,pipeline)

if __name__ == "__main__":
    print(data)