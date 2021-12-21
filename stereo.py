import numpy as np
from PIL import Image

class Stereo :
    '''
    Stereo 기본 베이스다. 이 구현에서는 left 이미지와 right 이미지를 알고 있다고 가정한다.
    물론 몰라도 된다. 이 구현에서는 몰라도 작동되게 되어있다.
    그렇지만 모른다면 search space가 매우 커진다.
    
    먼저 left image와 right image, kernel size를 초기화해준다.
    그리고 stereo 작동시 필요한 half of kernel size, row_size, col_size를 초기화해준다.

    출력해야되는 값인 disparity도 미리 준비해둔다.
    '''
    def __init__(self, left_img_path, right_img_path, kernel_size) :
        
        left_img = Image.open(left_img_path).convert('L')
        self.left_img = np.asarray(left_img)

        right_img = Image.open(right_img_path).convert('L')
        self.right_img = np.asarray(right_img)

        self.kernel_size = kernel_size
        self.kernel_size_half = int(self.kernel_size / 2)

        print(self.left_img.shape)
        self.row_size, self.col_size = self.left_img.shape 
        
        self.disparity = np.zeros((self.row_size, self.col_size), np.uint8)

    '''
    row를 기준으로 epipolar 라인을 설정한다. 그리고 row의 각 colunm을 중심으로 하는 패치가 있다고 생각한다.
    left 이미지의 patch를 따온다(center 값을 전달한다.)
    그리고 그 패치를 right 이미지의 패치와 비교한다.
    ssd가 가장 작을 때의 right image colunm 값과 지금 비교의 기준인 left image colunm 값을 뺀 patch라는 값을 disparity에 저장한다.
    '''
    def run(self) :
        for row in range(self.kernel_size_half, self.row_size - self.kernel_size_half):              
            print("\rProcessing.. %d%% complete"%(row / (self.row_size - self.kernel_size_half) * 100), end="", flush=True)        
            
            for col in range(self.kernel_size_half, self.col_size - self.kernel_size_half):
                min_ssd = 987654321
                best_patch = 0
                
                for patch in range(100):               
                    ssd = self.cal_ssd((row, col), patch)
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_patch = patch
                                
                self.disparity[row, col] = (best_patch / 100)* 255

    def save(self, file_name) :
        Image.fromarray(self.disparity).save(file_name)

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
    ncc를 계산합니다. ssd를 이용하는 것이 편하다.
    '''
    def cal_ncc(self, center, patch) :
        ncc = 0                           
        
        for u in range(-self.kernel_size_half, self.kernel_size_half):
            for v in range(-self.kernel_size_half, self.kernel_size_half):
                ncc += int(self.left_img[center[0]+u, center[1]+v]) * int(self.right_img[center[0]+u, (center[1]+v) - patch])  
                

        return 1 - ncc

if __name__ == '__main__':
    stereo = Stereo('img/im2.png', 'img/im6.png', 12)

    stereo.run()
    stereo.save('version_new3.png')