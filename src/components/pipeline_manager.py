import subprocess

class PipelineManager:
    """Manages and executes the data processing pipeline steps."""

    def __init__(self):
        self.steps = [
            "../pipeline/discard_non_valid.py",
            "../pipeline/discard_internal_redundant.py",
            "../pipeline/discard_external_redundant.py"
        ]

    def run_step(self, script_name):
        """Runs a Python script as a subprocess and prints the output."""
        print(f"🚀 Running {script_name}...")
        result = subprocess.run(["python", script_name], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully!")
            print(result.stdout)
        else:
            print(f"❌ Error in {script_name}:")
            print(result.stderr)

    def run_pipeline(self):
        """Executes all steps of the data processing pipeline in order."""
        for step in self.steps:
            self.run_step(step)

        print("🎉 Full pipeline executed successfully!")

# Run the pipeline
if __name__ == "__main__":
    manager = PipelineManager()
    manager.run_pipeline()
