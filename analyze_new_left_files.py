#!/usr/bin/env python3
"""
new_Left 폴더의 파일들을 분석하는 스크립트
Left_dummy 폴더는 제외하고 분석
"""

import os
from collections import defaultdict
from pathlib import Path

def analyze_folder(folder_path, exclude_dirs=None):
    """
    폴더의 파일들을 분석
    
    Args:
        folder_path: 분석할 폴더 경로
        exclude_dirs: 제외할 폴더명 리스트
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    # 파일 수집
    files = []
    values = []
    
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        
        # 제외 폴더 스킵
        if os.path.isdir(full_path):
            if entry in exclude_dirs:
                print(f"[제외됨] {entry} 폴더는 분석에서 제외됩니다.")
            continue
        
        # 이미지 파일만 처리
        if entry.lower().endswith(('.jpg', '.png', '.jpeg')):
            files.append(entry)
            
            # 파일명에서 첫 번째 언더스코어 전 숫자 추출
            try:
                num = int(entry.split('_')[0])
                values.append(num)
            except (ValueError, IndexError):
                pass
    
    return files, values


def print_distribution(values, bin_size=50):
    """50단위로 분포 출력"""
    if not values:
        print("분석할 파일이 없습니다.")
        return
    
    # 빈에 데이터 할당
    bins = defaultdict(int)
    for val in values:
        bin_idx = (val // bin_size) * bin_size
        bins[bin_idx] += 1
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"총 파일 수: {len(values)}")
    print(f"최소값: {min(values)}, 최대값: {max(values)}")
    print(f"\n{bin_size}단위 분포:")
    print(f"{'-'*60}")
    print(f"{'범위':<15} {'개수':<10} {'비율':<10} {'시각화'}")
    print(f"{'-'*60}")
    
    for bin_start in sorted(bins.keys()):
        bin_end = bin_start + (bin_size - 1)
        count = bins[bin_start]
        percent = (count / len(values)) * 100
        bar = "█" * max(1, int(count / 2))
        print(f"{bin_start:3d}-{bin_end:3d}     {count:6d}개   {percent:5.1f}%   {bar}")
    
    print(f"{'-'*60}")


def get_unique_values(values):
    """고유한 값들과 개수"""
    unique_vals = defaultdict(int)
    for val in values:
        unique_vals[val] += 1
    
    return unique_vals


def main():
    folder_path = "/abr/coss11/repo/new_data/new_Left"
    exclude_dirs = ["Left_dummy"]
    
    print(f"\n분석 경로: {folder_path}")
    print(f"제외 폴더: {exclude_dirs}")
    
    # 파일 분석
    files, values = analyze_folder(folder_path, exclude_dirs)
    
    # 분포 출력
    print_distribution(values, bin_size=50)
    
    # 상세 통계
    print(f"\n{'='*60}")
    print("상세 통계:")
    print(f"{'-'*60}")
    unique_vals = get_unique_values(values)
    print(f"고유한 값의 개수: {len(unique_vals)}")
    print(f"가장 많은 값: {max(unique_vals, key=unique_vals.get)} (출현: {max(unique_vals.values())}회)")
    print(f"가장 적은 값: {min(unique_vals, key=unique_vals.get)} (출현: {min(unique_vals.values())}회)")
    
    # 값 분포 (상위 20개)
    print(f"\n상위 20개 값의 분포:")
    print(f"{'-'*60}")
    sorted_vals = sorted(unique_vals.items(), key=lambda x: x[1], reverse=True)
    for idx, (val, count) in enumerate(sorted_vals[:20], 1):
        percent = (count / len(values)) * 100
        print(f"{idx:2d}. 값 {val:3d}: {count:4d}개 ({percent:5.1f}%)")


if __name__ == "__main__":
    main()
