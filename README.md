# SL.LSTM

các bước để thu data_v2

# 1
clone các mục cần thiết như là 

                              file Key_Point_v2.p để chạy file thu dữ liệu 

                              setup.txt để cài đặt các gói package cần dùng 
                              
                              folder Data_v2 để lưu data.
                              
# Key_Point_v2
cần lưu ý chính sửa các mục sau actions = np.array(['Ă']) để tùy chỉnh muốn thu data cho nhãn nào

no_sequences = 2   để chỉnh số lượng thao tác cho nhãn đó

sequence_length = 80  chỉnh số lượng frame cho nhãn

 -Folder start đặc biệt lưu ý file bắt đầu để tránh ghi đè lên file cũ

start_folder = 0

# phần tạo folder mới cần comment nếu k muốn sinh folder mới mỗi khi chạy code

      for action in actions:
          dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
          for sequence in range(1,no_sequences+1):
              try:
                  os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
              except:
                  pass

# output file
output_file = "video/test.mp4"

chỉnh sửa đường dẫn với tên phù hợp khi chạy thu data
# input file
DATA_PATH = os.path.join('Data_v2')
 nếu có lỗi thì chỉnh sửa đường dẫn lưu data tại đây
 
# 1 số lưu ý khác
khi tạo folder nhãn ví dụ là D thì cần tạo folder con bên trong tương ứng với start_folder 
ví dụ start_folder = 0 thì cần có folder 0 bên trong folder D
