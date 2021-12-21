import numpy as np
from PIL import Image
from enum import Enum

class Direction(Enum) :
    UP_LEFT = 1
    LEFT = 2
    UP = 3

class Stereo :
    '''
    Stereo DP를 수행힌다. 
    
    먼저 left image와 right image, kernel size를 초기화한다.
    그리고 stereo 작동시 필요한 half of kernel size, row_size, col_size를 초기화한다.

    상수 값으로 이용될 occlusion 값도 초기화한다.
    다이나믹 프로그래밍에 사용될 dp도 초기화한다.
    이동 값을 저장하는 move도 초기화한다.
    
    출력해야되는 값인 disparity도 미리 준비한다.
    '''
    def __init__(self, left_img_path, right_img_path, kernel_size, occulsion) :
        
        left_img = Image.open(left_img_path).convert('L')
        self.left_img = np.asarray(left_img)

        right_img = Image.open(right_img_path).convert('L')
        self.right_img = np.asarray(right_img)

        self.kernel_size = kernel_size
        self.kernel_size_half = int(self.kernel_size / 2)

        print(self.left_img.shape)
        self.row_size, self.col_size = self.left_img.shape 
        
        self.occulsion = occulsion

        self.disparity = np.zeros((self.row_size, self.col_size), np.uint8)

        self.dissimilarity = np.zeros((self.col_size, self.col_size), np.uint32)
        self.dp = np.zeros((self.col_size, self.col_size), np.uint32)
        self.move = np.zeros((self.col_size, self.col_size), np.uint8)
    '''
    row를 기준으로 epipolar 라인을 설정한다. 그리고 row의 각 colunm을 중심으로 하는 패치가 있다고 생각한다.
    left 이미지의 patch를 따온다(center 값을 전달)
    그리고 그 패치를 right 이미지의 패치와 비교한다.
    ssd를 dissimilarity 배열에 저장한다.

    그리고 dissimilarity 배열을 이용해 dp를 수행하고
    dp를 수행하면서 채운 move 배열을 이용해 disparity를 계산한다.
    '''
    def run(self) :
        for row in range(self.kernel_size_half, self.row_size - self.kernel_size_half):              
            print("\rProcessing.. %d%% complete"%(row / (self.row_size - self.kernel_size_half) * 100), end="", flush=True)        
            
            for col_left in range(self.kernel_size_half, self.col_size - self.kernel_size_half):
                for col_right in range(self.kernel_size_half, self.col_size - self.kernel_size_half):               
                    ssd = self.cal_ssd_by_centers((row, col_left), (row, col_right))
                    self.dissimilarity[col_left, col_right] = ssd
            
            self.cal_dp()
            self.cal_disparity_from_move(row)
            self.save('version_new3.png')

    def save(self, file_name) :
        Image.fromarray(self.disparity).save(file_name)

        disparity = (self.disparity / np.max(self.disparity)) * 255

        Image.fromarray(disparity.astype(np.uint8)).save("norm_" + file_name)

    '''
    ssd를 계산합니다. ssd를 이용하는 것이 편하다.
    ncc도 도전해보았는데 이미지를 [-1, 1]로 정규화하는 과정이 필요해 번거롭다.
    정규화 안하고 ncc를 사용하면...결과가 좋지 않았다.
    '''
    def cal_ssd(self, center, patch) :
        ssd = 0                           
        
        for u in range(-self.kernel_size_half, self.kernel_size_half):
            for v in range(-self.kernel_size_half, self.kernel_size_half):
                temp_ssd = int(self.left_img[center[0]+u, center[1]+v]) - int(self.right_img[center[0]+u, (center[1]+v) - patch])  
                ssd += (temp_ssd ** 2)

        return ssd

    '''
    left 이미지와 right 이미지의 center를 기준으로 ssd를 계산한다.
    '''
    def cal_ssd_by_centers(self, center_left, center_right) :
        ssd = 0                           
        
        for u in range(-self.kernel_size_half, self.kernel_size_half):
            for v in range(-self.kernel_size_half, self.kernel_size_half):
                temp_ssd = int(self.left_img[center_left[0]+u, center_left[1]+v]) - int(self.right_img[center_right[0]+u, (center_right[1]+v)])  
                ssd += (temp_ssd ** 2)

        return ssd
    
    '''
    ncc를 계산한다. ssd를 이용하는 것이 편하다.
    '''
    def cal_ncc(self, center, patch) :
        ncc = 0                           
        
        for u in range(-self.kernel_size_half, self.kernel_size_half):
            for v in range(-self.kernel_size_half, self.kernel_size_half):
                ncc += int(self.left_img[center[0]+u, center[1]+v]) * int(self.right_img[center[0]+u, (center[1]+v) - patch])  
                

        return 1-ncc

    '''
    dp를 계산한다. 외곽의 값을 먼저 채워준다.
    그리고 이 값들을 이용해 dp를 수행한다.
    cost가 minimize되는 최적의 경로를 저장한다.
    dp 공식은 직관적이라 이해하기 쉽다.
    '''
    def cal_dp(self) :

        for i in range(self.col_size - self.kernel_size_half) :
            self.dp[self.kernel_size_half-1][i] = i*self.occulsion
        for i in range(self.col_size - self.kernel_size_half) :
            self.dp[i][self.kernel_size_half-1] = i*self.occulsion

        for row in range(self.kernel_size_half, self.col_size - self.kernel_size_half) :
            for col in range(self.kernel_size_half, self.col_size - self.kernel_size_half) :
                from_up_left = self.dp[row-1][col-1] + self.dissimilarity[row][col]
                from_left = self.dp[row][col-1] + self.dissimilarity[row][col]
                from_up = self.dp[row-1][col] + self.dissimilarity[row][col]

                self.dp[row][col] = min(from_up_left, from_left, from_up)

                if(self.dp[row][col] == from_up_left) :
                    self.move[row][col] = Direction.UP_LEFT.value
                if(self.dp[row][col] == from_up) :
                    self.move[row][col] = Direction.UP.value
                if(self.dp[row][col] == from_left) :
                    self.move[row][col] = Direction.LEFT.value
    
    '''
    dp를 수행하면서 발생한 move 배열을 가지고 disparity를 계산한다.
    '''
    def cal_disparity_from_move(self, epipolar_line) :

        self.cal_disparity_from_move_impl(epipolar_line, self.col_size - self.kernel_size_half - 1, self.col_size - self.kernel_size_half - 1)
    
    '''
    재귀함수를 이용해 move 값을 트래킹한다.
    up_left 발생 시에만 pixel 위치차를 계산한다.
    '''
    def cal_disparity_from_move_impl(self, epipolar_line, row, col) :

        if(self.move[row][col] == Direction.UP.value) :
            return self.cal_disparity_from_move_impl(epipolar_line, row-1, col)
        if(self.move[row][col] == Direction.LEFT.value) :
            return self.cal_disparity_from_move_impl(epipolar_line, row, col-1)
        if(self.move[row][col] == Direction.UP_LEFT.value) :
            self.disparity[epipolar_line][row] = abs(row - col)
            return self.cal_disparity_from_move_impl(epipolar_line, row-1, col-1)

if __name__ == '__main__':
    stereo = Stereo('img/im2.png', 'img/im6.png', 12, 10)

    stereo.run()
    stereo.save('version_new3.png')