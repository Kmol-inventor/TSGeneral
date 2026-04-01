import sys
import os
import debugpy

# Add project root to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsgeneral import Inspector, Pipeline, StatefulPipeline
from tests.test_data.generators import generate_eeg_trials_by_samples

debugpy.listen(5678)  


data = generate_eeg_trials_by_samples(n_trials=10, n_samples=500, seed=42)

pipeline = Pipeline()
spip = StatefulPipeline()


pipeline.add_stage("Raw")
spip.add_stage("Raw")






inspector = Inspector(data,pipeline)

if __name__ == "__main__":
    inspector.run()