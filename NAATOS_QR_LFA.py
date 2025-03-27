import cv2
import filedialpy
from itertools import combinations
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from pybaselines import Baseline
from pyzbar.pyzbar import decode, ZBarSymbol
from qreader import QReader
from scipy.signal import savgol_filter,find_peaks

pd.set_option("display.precision", 2)

IMAGE_RESOLUTION_DPI = 300
PIXEL_PER_MM = IMAGE_RESOLUTION_DPI/25.4
QR_LAMINATE_THRESHOLD = 55
QR_CONDITION_THRESHOLD = 150
PEAK_WINDOW_MM = 1.0

class ScannerImage:
    '''Class definition for ScannerQRImage with jig for Test Consumable RevJ.2'''
    def __init__(self, path=None):
        self.qreader = QReader()
        if path is None:
            self.path = Path(filedialpy.openFile())
        else:
            self.path = Path(path)
        self.image = self.rotate_QR_image(cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED))
        self.qrs = self.detect_QR_image(self.image, self.qreader)
        self.annotated = self.image.copy()

    @staticmethod
    def rotate_QR_image(image):
        qrs = decode(image, symbols=[ZBarSymbol.QRCODE])
        match qrs[0].orientation:
            case 'RIGHT':
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            case 'DOWN':
                image = cv2.rotate(image, cv2.ROTATE_180)
            case 'LEFT':
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

    @staticmethod
    def detect_QR_image(image, qreader):
        data, qrs = qreader.detect_and_decode(image=image, return_detections=True)
        return [dict(qr,**{'data':value}) for value, qr in zip(data, qrs)]

    def create_test_list(self, dx_scale=8):
        dx = self.image.shape[1]/dx_scale
        qrs_y_sorted = sorted(self.qrs, key=lambda x: x['cxcy'][1])
        tests = [TestConsumable(c, b, a) for a, b, c in combinations(qrs_y_sorted, 3) if abs(a['cxcy'][0] - b['cxcy'][0]) <= dx and abs(b['cxcy'][0] - c['cxcy'][0]) <= dx]
        tests = sorted(tests, key=lambda x: x.laminate_QR['cxcy'][0])
        for qr in qrs_y_sorted:
            x1, y1, x2, y2 = qr['bbox_xyxy'].astype(int)
            self.annotated = cv2.rectangle(self.annotated,(x1, y1), (x2, y2), (0, 255, 0), 2)
            self.annotated = cv2.putText(self.annotated, qr['data'], (int(x1) - 20, int(y1) - 10),
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2
                                          )
        for i, test in enumerate(tests):
            self.annotated = cv2.rectangle(self.annotated, *test.get_laminate_pxRectangle(), (255, 0, 0), 2)
            self.annotated = cv2.rectangle(self.annotated, *test.get_lfa_pxRectangle(), (0, 0, 255), 2)
            self.annotated = cv2.putText(self.annotated, str(i+1),
                                         (test.get_laminate_pxRectangle()[0][0] + 10, test.get_laminate_pxRectangle()[0][1] + 30),
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
                                        )
        for test in tests:
            test.set_laminate_image(self.annotated)
            test.set_lfa_image(self.image)
        return tests

class TestConsumable:
    '''Class definition for TestConsumable RevJ.2'''
    laminate_mm_dims = np.array([29, 125])
    QR_mm_offset = np.array([14.5, 17])
    lfa_mm_offset = np.array([13.25, 26.5])
    lfa_mm_dims = np.array([2.5, 12])
    #TL, IPCL, FCL
    peaks_mm_dim = np.array([[14.5, 29.2], [14.5, 31.7], [14.5, 36.1]])

    def __init__(self, build_QR, condition_QR, laminate_QR) -> None:
        self.build_QR = build_QR
        self.condition_QR = condition_QR
        self.laminate_QR = laminate_QR
        self.laminate_image = None
        self.lfa_image = None
        self.result_image = None

    @staticmethod
    def _rectangle_to_subimage(image, rectangle) -> np.array:
        return image[int(rectangle[0][1]):int(rectangle[1][1]), int(rectangle[0][0]):int(rectangle[1][0])]

    @staticmethod
    def _mm_to_pixel(mm: np.array) -> np.array:
        return mm * PIXEL_PER_MM
    
    def get_origin_pxCorner(self) -> np.array:
        return self.laminate_QR['cxcy'] - self._mm_to_pixel(self.QR_mm_offset)
    
    def get_laminate_pxRectangle(self) -> np.array:
        x1y1 = self.get_origin_pxCorner()
        x2y2 = x1y1 + self._mm_to_pixel(self.laminate_mm_dims)
        return (x1y1.astype(int), x2y2.astype(int))
    
    def get_lfa_pxRectangle(self) -> np.array:
        x1y1 = self.get_origin_pxCorner() + self._mm_to_pixel(self.lfa_mm_offset)
        x2y2 = x1y1 + self._mm_to_pixel(self.lfa_mm_dims)
        return (x1y1.astype(int), x2y2.astype(int))
    
    def get_peak_intervals(self) -> np.array:
        x_values = self._mm_to_pixel(self.peaks_mm_dim)[:,1] \
                   + self.get_origin_pxCorner()[1] \
                   - self.get_lfa_pxRectangle()[0][1]
        return [[int(val - self._mm_to_pixel(PEAK_WINDOW_MM)), int(val + self._mm_to_pixel(PEAK_WINDOW_MM))] for val in x_values]
    
    def set_laminate_image(self, image) -> None:
        self.laminate_image = self._rectangle_to_subimage(image, self.get_laminate_pxRectangle())
        return

    def set_lfa_image(self, image) -> None:
        self.lfa_image = self._rectangle_to_subimage(image, self.get_lfa_pxRectangle())
        return

    def _find_lfa_peaks(self, line_profile):
        x_intervals = self.get_peak_intervals()
        n_background = sum([xs[0]-xs[1] for xs in x_intervals])
        filtered = savgol_filter(line_profile, 13, 2)
        # baselined = filtered - Baseline().modpoly(filtered, poly_order=3)[0]
        baselined = filtered
        lowest_length = np.clip(len(baselined)//2, 1, n_background)-1
        lowest = np.sort(baselined)[0:lowest_length]
        background = np.mean(lowest) + 3*np.std(lowest)
        peaks_X,_=find_peaks(baselined, height=background)
        peaks_Y=baselined[peaks_X]

        peaks_XY_max = [max([[X, Y] for (X,Y) in zip(peaks_X, peaks_Y) if X >= a and X <= b], key=lambda x:x[1], default=[None, None]) for a, b, in x_intervals]
        peaks_XY_max.append([None, background])

        peaks_X_by_location, peaks_Y_by_location = zip(*peaks_XY_max)
        return baselined, list(peaks_X_by_location), list(peaks_Y_by_location)

    def process_lfa_image(self, channel=2):
        # color channels in OpenCV are BGR so red channel is index 2
        assert channel in [0, 1, 2], 'Allowed color channels are 0 (blue), 1 (green), or 2 (red).'
        line = 255-np.mean(self.lfa_image[:,:,channel], axis=1)
        line_data = self._find_lfa_peaks(line)
        return line_data
    
    def generate_lfa_results(self, width=500, save_path=None):
        line, peaks_x, peaks_y = self.process_lfa_image()
        line = np.flip(line)
        peaks_x = [len(line) - px if px is not None else px for px in peaks_x]
        img = cv2.flip(cv2.rotate(self.lfa_image, cv2.ROTATE_180), 1)
        x_intervals = [[len(line) - x for x in xs] for xs in self.get_peak_intervals()]
        w1=img.shape[1]
        w2=max(line)-min(line)+20
        col_widths = [w1/(w1+w2), w2/(w1+w2)]
    
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, specs = [[{}, {}]],
                                horizontal_spacing = 0.0, column_widths=col_widths)
        for xs in x_intervals:
            fig.add_trace(go.Scatter(x=[0, 255, 255, 0],
                                     y=np.repeat(xs, 2),
                                     fill='toself', 
                                     fillcolor='rgba(0,0,0,0.2)',
                                     line_color='rgba(0,0,0,0)',
                                     showlegend=False
            ), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=line,
                       mode='lines', 
                       name='width-avg intensity',
                       line=dict(color='red')
            ), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=peaks_x[:3], x=peaks_y[:3],
                       mode='markers',
                       name='peaks',
                       line=dict(color='red'),
                       showlegend=False
            ), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=[0,len(line)],
                       x=np.tile(peaks_y[3],2),
                       mode='lines', 
                       name='avg background',
                       line=dict(color='red', dash='dash')
            ), row=1, col=2
        )
        fig.add_trace(
            go.Image(z=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), row=1, col=1)
        fig.update_yaxes(range=[0, len(line)])
        fig.update_layout(autosize=False,
                          width=width,
                          xaxis2={'range': (min(line)-10, max(line)+10)},
                          margin=dict(l=0, r=20, t=20, b=25),
                          legend=dict(yanchor='top', xanchor='right', y=1, x=1)
        )
        b1 = self._image_to_nparray(fig, width)

        if peaks_y[0] is None:
            test_call = '<b>INVALID</b>'
        elif peaks_y[1] is None and peaks_y[2] is None:
            test_call = '<b>INVALID</b>'
        elif peaks_y[1] is not None and peaks_y[2] is None:
            test_call = '<b>NEGATIVE</b>'
        elif peaks_y[2] is not None:
            test_call = '<b>POSITIVE</b>'
        else:
            test_call = '<b>?</b>'
        table_peaks = peaks_y + ['']
        table_calls = ['<b>NOT DETECTED</b>' if peak_y is None else '<b>DETECTED</b>' for peak_y in peaks_y[:3]] + ['', test_call]
        table_labels = ['Flow Control Line', 'IPC Line', 'Test Line', 'Background (µ + 3σ)', self.condition_QR['data']]
        table_df = pd.DataFrame({'region': table_labels,
                    'peak intensity value': table_peaks,
                    'algorithm call': table_calls
        })
        if save_path is not None:
            output_peaks = peaks_y[:3]
            output_calls = [n for n in table_calls if n != '']
            output_labels = ['FC intensity', 'IPC intensity', 'TL intensity', 'FC call', 'IPC call', 'TL call', 'test call']
            output_df = pd.DataFrame({self.condition_QR['data']: output_peaks + output_calls},
                                    index=output_labels
                                    )
            csv_save_path = save_path.parent / f'{save_path.parts[-2]}.csv'
            if csv_save_path.exists():
                input_df = pd.read_csv(str(csv_save_path), index_col=0)
                pd.concat([input_df, output_df], axis=1).to_csv(str(csv_save_path))
            else:
                output_df.to_csv(str(csv_save_path))
        fig = ff.create_table(table_df)
        fig.update_layout(
            autosize=False,
            width=500,
            height=200,
        )
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 14
            if fig.layout.annotations[i]['text'] == '<b>DETECTED</b>':
                fig.layout.annotations[i].font.color = 'green'
            elif fig.layout.annotations[i]['text'] == '<b>NOT DETECTED</b>':
                fig.layout.annotations[i].font.color = 'red'
        b2 = self._image_to_nparray(fig, width)
        b = np.concatenate((b1, b2), axis=0)
        (h, w) = self.laminate_image.shape[:2]
        a = cv2.resize(self.laminate_image, (int(b.shape[0]*w/h), b.shape[0]), interpolation=cv2.INTER_AREA)
        return np.concatenate((a, b), axis=1)
    
    def _image_to_nparray(self, fig, width):
        img_str = fig.to_image(format='png', width=width)
        arr = np.frombuffer(img_str, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def show_cv2_image(image, scale=0.5, title=None):
    if title is None:
        title = f'image 1'
    cv2.imshow(title, cv2.resize(image, None, fx=scale, fy=scale))
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(title, 0, 0)
    while True:
        k = cv2.waitKey(100)
        if k >= 0:
            cv2.destroyAllWindows()
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:        
            break
    cv2.destroyAllWindows()

def show_cv2_images(images, scales=None, titles=None, save_paths=None):
    if titles is None:
        titles = [f'image {i+1}' for i in range(len(images))]
    if scales is None:
        scales = [1]*len(images)
    for i in range(len(images)):
        img = cv2.resize(images[i], None, fx=scales[i], fy=scales[i])
        if save_paths[i] is not None:
            img_save_path = save_paths[i].parent / f'{save_paths[i].stem} - {titles[i]}.png'
            cv2.imwrite(img_save_path, img)
        cv2.imshow(titles[i], img)
        cv2.setWindowProperty(titles[i], cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(titles[i], i*300, 0)
    windows_open = True
    while windows_open:
        k = cv2.waitKey(100)
        if k >= 0:
            cv2.destroyAllWindows()
            windows_open = False
        if [cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) for title in titles] == [0]*len(titles):
            windows_open = False
    cv2.destroyAllWindows()

scanned = ScannerImage()
tests = scanned.create_test_list()
n_tests = len(tests)
print(f'{n_tests} test consumables identified in the image at {scanned.path}.')
if True:
    show_cv2_images([test.generate_lfa_results(save_path=scanned.path) for test in tests], scales=[1]*n_tests, save_paths=[scanned.path]*n_tests)