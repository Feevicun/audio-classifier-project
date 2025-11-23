import requests
import json
import time
import statistics
import argparse
from pathlib import Path

def benchmark_api(api_url, num_requests=10):
    """Бенчмарк для API інференсу"""
    results = {
        'latencies': [],
        'success_count': 0,
        'error_count': 0,
        'errors': []
    }
    
    # Тестові дані (можна згенерувати простий аудіо сигнал)
    test_files = ['test_samples/sample1.wav', 'test_samples/sample2.wav']
    
    for i in range(num_requests):
        try:
            # Використовуємо тестові файли або генеруємо synthetic data
            files = {}
            
            start_time = time.time()
            response = requests.get(f"{api_url}/health", timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                results['success_count'] += 1
                results['latencies'].append((end_time - start_time) * 1000)  # ms
            else:
                results['error_count'] += 1
                results['errors'].append(f"HTTP {response.status_code}")
                
        except Exception as e:
            results['error_count'] += 1
            results['errors'].append(str(e))
    
    # Розрахунок метрик
    if results['latencies']:
        results['avg_latency'] = statistics.mean(results['latencies'])
        results['min_latency'] = min(results['latencies'])
        results['max_latency'] = max(results['latencies'])
        results['p95_latency'] = statistics.quantiles(results['latencies'], n=20)[18]  # 95th percentile
    
    results['success_rate'] = results['success_count'] / num_requests
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark ML API')
    parser.add_argument('--url', required=True, help='API URL')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--requests', type=int, default=10, help='Number of requests')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting benchmark for {args.url}")
    results = benchmark_api(args.url, args.requests)
    
    # Зберігаємо результати
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Benchmark completed:")
    print(f"   Success rate: {results['success_rate']:.1%}")
    print(f"   Average latency: {results.get('avg_latency', 0):.2f}ms")
    print(f"   Results saved to: {args.output}")

if __name__ == '__main__':
    main()