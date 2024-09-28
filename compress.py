import gzip
import shutil

# Compress the model file
def compress_model(model_filename):
    with open(model_filename, 'rb') as f_in:
        with gzip.open(model_filename + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Model compressed and saved as '{model_filename}.gz'")

# Example usage
compress_model('option_price_model.pkl')
