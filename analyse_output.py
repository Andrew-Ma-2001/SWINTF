import re
import csv
import argparse
import os

def parse_results(file_path):
    results = {}
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all matches using regex
    pattern = r'--config .*?/noise/(.+?)\.yaml.*?\n-- Average PSNR_Y/SSIM_Y: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    # Store results
    for noise_type, psnr in matches:
        results[noise_type] = float(psnr)
    
    return results

def write_csv(results, output_file='noise_results.csv'):
    # Prepare data for CSV
    headers = ['Noise Type', 'PSNR']
    rows = [[k, v] for k, v in results.items()]
    
    # Sort rows by noise type for better readability
    rows.sort(key=lambda x: x[0])
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze noise test results and generate CSV')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to the input text file containing test results')
    parser.add_argument('--output_file', type=str,
                       help='Path to save the output CSV file (default: input_filename.csv)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output_file is None:
        # Get the input filename without extension
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}.csv"
    
    # Parse the results file
    results = parse_results(args.input_file)
    
    # Write results to CSV
    write_csv(results, args.output_file)
    
    # Print results to verify
    print(f"Results written to {args.output_file}")
    print("\nParsed Results:")
    for noise_type, psnr in sorted(results.items()):
        print(f"{noise_type}: {psnr:.2f}")

if __name__ == "__main__":
    main()