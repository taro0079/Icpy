#-------------------------------------------------
# SuperCon_IV.py ver1.2
#
# Copyright (c) 2021, Data PlatForm Center, NIMS
# fixing by Taro Morita
# This software is released under the MIT License.
#-------------------------------------------------
# -*- coding: utf-8 -*-


"""
機能：超電導のI-V特性のデータ構造化および可視化
機器：無冷媒低磁場物性測定装置 IV特性＠桜標準実験棟335

1) 半導体パラメーター（IV測定）から測定時間，電流，電圧のデータがtxtとして出力されるファイルを読み込む．
2) ヘッダーから電圧端子間距離を取得し，電界強度を計算し，I-V特性図を描画する．
3) n値を自動的に計算する
4) Ic-B(T)図を作図する．

"""

__author__ = "Shigeyuki Matsunami"
__contact__ = "Matsunami.shigeyuki@nims.go.jp"
__license__ = "MIT"
__copyright__ = "National Institute for Materials Science, Japan"
__date__ = "2021/03/25"

# モジュール
import os
import glob
from natsort import natsorted
import re

# 数値処理用
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 可視化
from matplotlib import pyplot as plt

# util関数
# 実行処理#rawデータの読み込み


class Datafilerepo():
    def __init__(self, path):
        self._path = path
        self._filelst = natsorted(glob.glob(path))
        self._numfile = len(self._filelst)
        self._outfilename = [os.path.splitext(os.path.basename(p))[
            0] for p in self._filelst]

    def getPath(self):
        return self._path

    def getFileLst(self):
        return self._filelst

    def getNumFile(self):
        return self._numfile

    def getOutFileName(self):
        return self._outfilename

    def readlines(self):
        f = open(self._filelst[0])
        lines = f.readlines()
        return lines

    def getHeader(self):
        return self.readlines()[0].split()

    def splitlines(self, lines, header=2):
        return [i.split() for i in lines][header:-1]

    def cvttoFloat(self, lines):
        return [map(float, i) for i in lines]

    def cvttoDataFrame(self, data):
        return pd.DataFrame(data)

    def read(self):
        lines = self.readlines()
        splitlines = self.splitlines(lines)
        floatlines = self.cvttoFloat(splitlines)
        df = self.cvttoDataFrame(floatlines)
        df.columns = self.getHeader()
        return Data(df)


class Data():
    def __init__(self, data):
        self._current = data.iloc[:, 0].values.reshape(-1, 1)
        self._voltage = data.iloc[:, 1].values.reshape(-1, 1)

    def getCurrent(self):
        return self._current

    def getVoltage(self):
        return self._voltage


class Plots():
    def __init__(self, figsize=(6, 6)):
        self.fig = plt.figure(figsize=figsize)

    def simplePlot(self, voltage, current):
        ax1 = self.fig.add_subplot(111)
        ax1.set_xlabel("Current (A)")
        ax1.set_ylabel("Voltage (uV)")
        ax1.plot(voltage, current)
        plt.show()


if __name__ == '__main__':
    datainfo = Datafilerepo("./testdata/*")
    data = datainfo.read()
    p = Plots()
    p.simplePlot(data.getCurrent(), data.getVoltage())


    

def read_files(extension):

    # 読み込みファイルの取得
    # 拡張子がないため，"data"フォルダーに格納されたrawファイルを読み出すことにする
    path = 'data/*' + extension
    input_files = natsorted(glob.glob(path))

    # 読み込みファイル数の取得
    number_of_files = len(input_files)

    # 拡張子を抜いたファイル名（出力用）
    output_name = [os.path.splitext(os.path.basename(p))[0] for p in input_files]
    
    print(output_name)
    return input_files, number_of_files, output_name

def data_extract(file):

    # ファイルの読み込みとヘッダー部の読み出し
    df_header = pd.read_csv(file, header=None,nrows=10, delimiter=';') 
    
    #　端子間距離の取得
    VV_length = df_header[1][1]

    #M　磁場強度の取得
    text = df_header[0][4]

    # 出力がTypeI型の場合
    
    if "Time" in text:
        pattern = "(\d\.\d{6}\d*\.\d{6})Time" 
        pattern2 = "NaN(\d*\.\d{6})Time"

    
        if "NaN" in text:
            field = float(re.search(pattern2,text).group().replace('Time', '')[3:])
        else:
            field = float(re.search(pattern,text).group().replace('Time', '')[8:])
        
        header = 5


    #出力がType II型の場合    
    else:
        text = df_header[0][7]
        pattern = "(\d*\.\d{6})Time"
        field = float(re.search(pattern,text).group().replace('Time', ''))
        
        header = 8
        
    #　数値部の取り出し
    df_val = pd.read_csv(file,
                 header= header, 
                 delim_whitespace=True, 
                 names=('time', 'current', 'voltage','Temp.1','Temp.2','Temp.3'))
    
    #　電界強度の追加    
    df_val['Electric_field_strength'] = df_val['voltage']/float(VV_length)
    
    #　I-Vからのn値の計算
    n_value = get_N_value(df_val)
    
    return df_val, field, n_value

def make_IV(df,file):

    #　図の設定 
    hfont = {'fontname': 'Arial'}
    fig, ax = plt.subplots(1,1, figsize=(7,7))
             
    X = df['current']
    Y = df['Electric_field_strength'] 
                
    ax.plot(X,Y,c='blue')

     #　掃引速度の計算
    sweep_time = df.iloc[-1][1]/df.iloc[-1][0]

    #　作図のデザイン
    ax.set_xlabel('Current [A]',**hfont, fontsize = 18)
    ax.set_ylabel('Voltage [uV/cm]',**hfont, fontsize = 18)
    #ax.set_xlim(0,2)
    ax.set_ylim(0,100)
    ax.tick_params(direction = "inout", length = 5, labelsize=14)
    ax.text(0.01,90,'Sweep Rate: {:.3f} A/sec'.format(sweep_time), **hfont, fontsize = 16)
    ax.set_title(file,**hfont, fontsize = 16)
    ax.grid(which = "major", axis = "both", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)

    # y軸に目盛線を設定
    #ax.grid(which = "major", axis = "y", color = "blue", alpha = 0.8,linestyle = "--", linewidth = 0.1)
    
    #出力
    plt.savefig(file + '_I-V.png', dpi=300)

def make_n_value(df,file):

    #　図の設定
    hfont = {'fontname': 'Arial'}
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    
    # 電圧領域として1～10V/cm範囲での電流電圧特性からn値を算出
   
    index_min = getNearestValue(df['voltage'], 1)
    index_max = getNearestValue(df['voltage'], 10)

    Ic = df['current'][index_min:index_max]
    Vc = df['voltage'][index_min:index_max]
    
    x = np.log(Ic,dtype = float).reshape(-1, 1)
    y = np.log(Vc,dtype = float)
    
    n = get_N_value(df)
    
    ax.plot(x,y,c='blue',marker="o",linestyle='--')

    #　作図のデザイン
    ax.set_xlabel('log (current) [A]',**hfont, fontsize = 18)
    ax.set_ylabel('log (voltage) [uV]',**hfont, fontsize = 18)
    ax.tick_params(direction = "inout", which = "both", length = 5, labelsize=14)
    ax.text(min(x),max(y)-0.1,'n value: {} '.format(n), **hfont, fontsize = 16)
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
    ax.grid(which = "major", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
    ax.set_title(file,**hfont, fontsize = 16)
    #ax.set_xlim(4,12)
    #ax.set_ylim(0.001,100)
    
    #出力
    plt.savefig(file + '_n-value.png', dpi=300)

    
def make_Ic_B(df):

    #図の設定
    hfont = {'fontname': 'Arial'}
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    
            
    X = df['Magnetic_Field']
    Y = df['Ic'] 
         
    ax.plot(X,Y,c='blue',marker="o",linestyle='--')

    #作図のデザイン
    ax.set_yscale('log')
    ax.set_xlabel('B [T]',**hfont, fontsize = 18)
    ax.set_ylabel('Critical Current [A]',**hfont, fontsize = 18)
    ax.tick_params(direction = "inout", which = "both", length = 5, labelsize=14)
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
    ax.grid(which = "minor", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)

    ax.set_xlim(4,12)
    ax.set_ylim(0.001,100)
    ax.set_title('Ic - B(T)',**hfont, fontsize = 16)
    
    #出力
    plt.savefig('Ic-B.png', dpi=300)
    
def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return idx

def get_N_value(df):
    """
    概要: I-V特性からn値を算出する
    min: 電圧0.1Vのインデックスを取得
    max：電圧10Vのインデックスを取得
    return：log-log値の傾きをLinearRegressionで算出．その傾きをreturnする
    """
    
    # 電圧領域として1～10V/cm範囲での電流電圧特性からn値を算出
    index_min = getNearestValue(df['voltage'], 1)
    index_max = getNearestValue(df['voltage'], 10)

    Ic = df['current'][index_min:index_max]
    Vc = df['voltage'][index_min:index_max]
    
    x = np.log(Ic,dtype = float).reshape(-1, 1)
    y = np.log(Vc,dtype = float)
    
    solv = LinearRegression()
    solv.fit(x,y)
    
    return format(*solv.coef_, '.2f') 

# 実行処理
#print(val.current)
#def main():
#        
#    # データの取得
#    #　rawファイルは拡張子がない
#    default_extension = ''
#
#    # ファイルの読み込み
#    [files, f_num, fname] = read_files(default_extension)
#    
#    #　磁場，臨界電流，n値の初期設定
#    Magnetic_Field = []
#    Ic = []
#    n_value = []
#
#    #　I-V特性のマルチプロットの初期設定
#    hfont = {'fontname': 'Arial'}
#    fig, ax = plt.subplots(1,1, figsize=(7,7))
#    cmap = plt.get_cmap("tab10")
#    
#    
#    # データ抽出（分割された複数ファイル）
#    for i in range(f_num):
#        
#        try:
#            #　数値データ部，磁場，n値の取得
#            data,mf,n = data_extract(files[i])        
#            Magnetic_Field.append(mf)
#            n_value.append(n)
#
#            #　臨界電流値の取得
#            index = getNearestValue(data['Electric_field_strength'], 1)
#            Ic.append(data['current'][index])
#        
#            #　数値データのcsv出力
#            data.to_csv(fname[i]+'_extract.csv')
#
#            #　可視化
#            make_IV(data, fname[i])   
#            make_n_value(data, fname[i])
#        
#        except:
#            continue
#
#        
#        #マルチプロット
#        X = data['current']
#        Y = data['Electric_field_strength'] 
#        
#        ax.plot(X,Y,color=cmap(i),label = fname[i])
#    
#    ax.set_xlabel('Current [A]',**hfont, fontsize = 18)
#    ax.set_ylabel('Voltage [uV/cm]',**hfont, fontsize = 18)
#    ax.set_ylim(0,80)
#    ax.grid(which = "major", axis = "both", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
#
#    ax.tick_params(direction = "inout", length = 5, labelsize=14)
#    ax.set_title('I-V-all',**hfont, fontsize = 16)
#    ax.legend()
#    fig.savefig('I-V-all.png', dpi=300)
#    
#    # Ic-Bの作成と出力
#    df_MF = pd.DataFrame(Magnetic_Field, columns = ['Magnetic_Field'])
#    df_Ic = pd.DataFrame(Ic,columns = ['Ic'])
#    df_n = pd.DataFrame(n_value,columns = ['n_value'])
#    df = pd.concat([df_MF,df_Ic,df_n],axis=1)
#    
#    make_Ic_B(df)
#    
#    df.to_csv('Ic-MF-n.csv',index = False)
        
#if __name__ == '__main__':
#    main()
