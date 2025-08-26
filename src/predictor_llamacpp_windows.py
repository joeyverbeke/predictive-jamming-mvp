import os
import platform
from typing import Optional
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not available")


class PredictorLlamaCPPWindows:
    """Windows-optimized next-token predictor using llama-cpp-python with GPU support"""
    
    def __init__(self, model_path="models/llm/llama-3.2-1b-q4_k_m.gguf", context_tokens=96, 
                 gpu_acceleration=True, gpu_layers=32):
        """
        Initialize Windows LLM predictor with GPU support
        
        Args:
            model_path: Path to GGUF model file (Windows path)
            context_tokens: Maximum context tokens to use
            gpu_acceleration: Enable GPU acceleration for RTX 3060
            gpu_layers: Number of layers to offload to GPU
        """
        self.context_tokens = context_tokens
        self.gpu_acceleration = gpu_acceleration
        self.gpu_layers = gpu_layers
        
        # Convert to Windows path if needed
        if platform.system() == "Windows":
            model_path = str(Path(model_path))
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model not found at {model_path}")
        
        # Load LLM model with Windows-specific settings
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama-cpp-python not available. Please install with GPU support.")
        
        try:
            # Windows-specific model loading with GPU support
            model_kwargs = {
                'model_path': model_path,
                'n_ctx': 1024,
                'logits_all': False,
                'verbose': False,
            }
            
            # Add GPU acceleration if enabled
            if self.gpu_acceleration and platform.system() == "Windows":
                try:
                    # Check if CUDA is available
                    import torch
                    if torch.cuda.is_available():
                        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                        model_kwargs.update({
                            'n_gpu_layers': self.gpu_layers,
                            'n_batch': 512,  # Optimize for RTX 3060
                            'use_mmap': True,
                            'use_mlock': False,  # Windows compatibility
                        })
                        print(f"GPU acceleration enabled with {self.gpu_layers} layers")
                    else:
                        print("CUDA not available, using CPU")
                        model_kwargs['n_gpu_layers'] = 0
                except ImportError:
                    print("PyTorch not available, using CPU")
                    model_kwargs['n_gpu_layers'] = 0
            else:
                model_kwargs['n_gpu_layers'] = 0
            
            self.model = Llama(**model_kwargs)
            print(f"Loaded LLM model from {model_path}")
            
            # Log GPU usage
            if self.gpu_acceleration and hasattr(self.model, 'n_gpu_layers'):
                if self.model.n_gpu_layers > 0:
                    print(f"Model loaded with {self.model.n_gpu_layers} GPU layers")
                else:
                    print("Model loaded with CPU only")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model: {e}")
    
    def predict_horizon(self, rolling_text: str, params: dict) -> str:
        """
        Predict next tokens (horizon) with Windows optimizations
        
        Args:
            rolling_text: Current text context
            params: Prediction parameters (min_pred_tokens, max_pred_tokens, top_k)
            
        Returns:
            Predicted continuation as lowercase string
        """
        if not rolling_text.strip():
            return ""
        
        # Truncate to last context_tokens by simple tail truncation
        words = rolling_text.split()
        if len(words) > self.context_tokens:
            truncated_text = " ".join(words[-self.context_tokens:])
        else:
            truncated_text = rolling_text
        
        # Prepare prompt
        prompt = truncated_text.strip()
        
        try:
            # Windows-optimized generation parameters
            generation_kwargs = {
                'max_tokens': params.get('max_pred_tokens', 5),
                'temperature': 0.0,
                'top_k': params.get('top_k', 20),
                'stop': ["\n"],
                'echo': False,
            }
            
            # Add Windows-specific optimizations
            if platform.system() == "Windows":
                generation_kwargs.update({
                    'repeat_penalty': 1.1,  # Prevent repetition
                    'top_p': 0.9,  # Nucleus sampling
                })
            
            # Generate prediction
            response = self.model(
                prompt,
                **generation_kwargs
            )
            
            # Extract generated text
            generated_text = response['choices'][0]['text'].strip()
            
            # Return lowercase prediction
            return generated_text.lower()
            
        except Exception as e:
            print(f"LLM prediction error: {e}")
            return ""
    
    def test_prediction(self, text: str) -> str:
        """
        Test prediction with default parameters
        
        Args:
            text: Input text
            
        Returns:
            Predicted continuation
        """
        params = {
            'min_pred_tokens': 3,
            'max_pred_tokens': 5,
            'top_k': 20
        }
        return self.predict_horizon(text, params)
    
    def benchmark_performance(self, test_text: str = "The quick brown fox jumps over the lazy dog", 
                            iterations: int = 10) -> dict:
        """
        Benchmark LLM performance on Windows
        
        Args:
            test_text: Text to use for benchmarking
            iterations: Number of iterations to run
            
        Returns:
            Performance metrics
        """
        import time
        
        print(f"Benchmarking LLM performance with {iterations} iterations...")
        
        # Warm up
        for _ in range(3):
            self.test_prediction(test_text)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            self.test_prediction(test_text)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        metrics = {
            'total_time': total_time,
            'avg_time_per_prediction': avg_time,
            'predictions_per_second': iterations / total_time,
            'gpu_acceleration': self.gpu_acceleration,
            'gpu_layers': self.gpu_layers if self.gpu_acceleration else 0
        }
        
        print(f"Benchmark results:")
        print(f"  Average time per prediction: {avg_time:.3f}s")
        print(f"  Predictions per second: {metrics['predictions_per_second']:.1f}")
        print(f"  GPU acceleration: {self.gpu_acceleration}")
        
        return metrics


def check_gpu_availability() -> dict:
    """Check GPU availability and capabilities on Windows"""
    gpu_info = {
        'cuda_available': False,
        'gpu_name': None,
        'gpu_memory': None,
        'llama_cpp_gpu': False
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            print(f"CUDA GPU detected: {gpu_info['gpu_name']}")
            print(f"GPU Memory: {gpu_info['gpu_memory']:.1f} GB")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not available")
    
    # Check if llama-cpp-python has GPU support
    if LLAMA_AVAILABLE:
        try:
            # Try to create a minimal model to test GPU support
            test_model = Llama(
                model_path="dummy",  # This will fail, but we can check if GPU args are accepted
                n_gpu_layers=1
            )
            gpu_info['llama_cpp_gpu'] = True
        except:
            gpu_info['llama_cpp_gpu'] = False
    
    return gpu_info


if __name__ == "__main__":
    # Test GPU availability
    print("Windows LLM Predictor GPU Check")
    print("=" * 40)
    
    gpu_info = check_gpu_availability()
    
    print(f"\nGPU Summary:")
    print(f"  CUDA Available: {gpu_info['cuda_available']}")
    print(f"  GPU Name: {gpu_info['gpu_name']}")
    print(f"  GPU Memory: {gpu_info['gpu_memory']:.1f} GB" if gpu_info['gpu_memory'] else "  GPU Memory: Unknown")
    print(f"  Llama-cpp GPU Support: {gpu_info['llama_cpp_gpu']}")
    
    # Test model loading if available
    model_path = "models/llm/llama-3.2-1b-q4_k_m.gguf"
    if os.path.exists(model_path):
        print(f"\nTesting model loading...")
        try:
            predictor = PredictorLlamaCPPWindows(
                model_path=model_path,
                gpu_acceleration=gpu_info['cuda_available']
            )
            
            # Test prediction
            test_text = "The weather is"
            prediction = predictor.test_prediction(test_text)
            print(f"Test prediction: '{test_text}' -> '{prediction}'")
            
            # Benchmark
            predictor.benchmark_performance()
            
        except Exception as e:
            print(f"Model test failed: {e}")
    else:
        print(f"\nModel not found at {model_path}")
        print("Please download the model first.")
