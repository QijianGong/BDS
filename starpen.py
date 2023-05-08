#mPythonType:0
from machine import UART,Timer,RTC
from mpython import *
import math

LON = 113.99194 #观测者经度
LAT = 22.58750  #观测者纬度
d   = 8034.5    #距离J2000的天数
UT  = 0
UTC = "2000/01/01 00:00"
days = [0,31,59,90,120,151,181,212,243,273,304,334]
init = 0
tim1 = Timer(1) #测试间隔
Xmax,Xmin,Ymax,Ymin,Zmax,Zmin = 1,-1,1,-1,1,-1 #磁力校准数据

STAR=[[0,101.287156,-16.716114,'天狼星','大犬座','43'],[1,95.987959,-52.695659,'老人','船底座','34'],[2,219.902063,-60.833973,'南门二','半人马座','09'],[3,279.234736,38.783693,'织女星','天琴座','52'],[4,213.915302,19.182414,'大角','牧夫座','13'],
[5,79.17233,45.997992,'五车二','御夫座','21'],[6,78.634583,-8.201667,'参宿七','猎户座','26'],[7,114.825493,5.224994,'南河三','小犬座','71'],[8,24.428749,-57.236668,'水委一','波江座','06'],[9,88.792939,7.407063,'参宿四','猎户座','26'],
[10,210.955853,-60.373038,'马腹一','半人马座','09'],[11,297.69583,8.868323,'牛郎星','天鹰座','22'],[12,68.980162,16.509302,'毕宿五','金牛座','17'],[13,247.351921,-26.432001,'心宿二','天蝎座','33'],[14,201.298248,-11.16132,'角宿一','室女座','02'],
[15,116.328961,28.026199,'北河三','双子座','30'],[16,344.412695,-29.622235,'北落师门','南鱼座','60'],[17,191.930263,-59.688762,'十字架三','南十字座','88'],[18,310.357978,45.280339,'天津四','天鹅座','16'],[19,186.649567,-63.09909,'十字架二','南十字座','88'],
[20,152.092962,11.967208,'轩辕十四','狮子座','12'],[21,104.656250,-28.972221,'弧矢七','大犬座','43'],[22,113.649429,31.888277,'北河二','双子座','30'],[23,263.402167,-37.10382,'尾宿八','天蝎座','33'],[24,81.282763,6.349703,'参宿五','猎户座','26'],
[25,81.572973,28.607451,'五车五','金牛座','17'],[26,84.053337,-1.201944,'参宿二','猎户座','26'],[27,332.058273,-46.960974,'鹤一','天鹤座','45'],[28,193.50729,55.959822,'玉衡','大熊座','03'],[29,122.383127,-47.336586,'天社一','船帆座','32'],
[30,51.080711,49.86118,'天船三','英仙座','24'],[31,165.931954,61.751034,'天枢','大熊座','03'],[32,107.097859,-26.393207,'弧矢一','大犬座','43'],[33,276.042993,-34.384615,'箕宿三','人马座','15'],[34,206.885157,49.313266,'摇光','大熊座','03'],
[35,37.954516,89.26411,'北极星','小熊座','56'],[36,252.16623,-69.027713,'三角形三','南三角座','83'],[37,306.411908,-56.735088,'孔雀十一','孔雀座','44'],[38,141.896851,-8.6586,'星宿一','长蛇座','01'],[39,31.793363,23.462424,'娄宿三','白羊座','39'],
[40,10.896822,-17.986677,'土司空','鲸鱼座','04'],[41,17.433016,35.620558,'奎宿九','仙女座','19'],[42,263.733628,12.560035,'侯','蛇夫座','11'],[43,233.671951,26.714694,'贯索四','北冕座','73'],[44,269.151542,51.488896,'天棓四','天龙座','08'],
[45,10.126836,56.537332,'王良四','仙后座','25'],[46,120.896028,-40.003146,'弧矢增二十二','船尾座','20'],[47,220.482316,-47.388199,'骑官十','豺狼座','46'],[48,345.943573,28.08279,'室宿二','飞马座','07'],[49,6.570942,-42.306077,'火鸟六','凤凰座','37'],
[50,319.644126,62.585462,'天钩五','仙王座','27'],[51,183.951925,-17.54198,'轸宿一','乌鸦座','70'],[52,83.182557,-17.822291,'厕一','天兔座','51'],[53,229.251956,-9.382867,'氐宿四','天秤座','29'],[54,236.066979,6.425628,'天市右垣七','巨蛇座','23'],
[55,84.912252,-34.074052,'丈人一','天鸽座','54'],[56,189.295987,-69.135612,'蜜蜂三','苍蝇座','77'],[57,250.321502,31.602726,'天纪二','武仙座','05'],[58,6.437917,-77.254166,'蛇尾一','水蛇座','61'],[59,261.324954,-55.529883,'杵三','天坛座','63'],
[60,326.760188,-16.127284,'垒壁阵四','摩羯座','40'],[61,334.625393,-60.259586,'鸟喙一','杜鹃座','48'],[62,322.889726,-5.571171,'虚宿一','宝瓶座','10'],[63,194.006948,38.31838,'常陈一','猎犬座','38'],[64,32.385944,34.987295,'天大将军九','三角座','78'],
[65,309.391801,-47.2915,'波斯二','印第安座','49'],[66,140.263756,34.392563,'轩辕四','天猫座','28'],[67,220.626753,-64.975136,'南门增二','圆规座','85'],[68,68.498856,-55.045022,'金鱼二','剑鱼座','72'],[69,102.047727,-61.94132,'金鱼增一','绘架座','59'],
[70,63.606183,-62.473859,'夹白二','网罟座','82'],[71,299.689282,19.492148,'左旗五','天箭座','86'],[72,276.743402,-45.968457,'鳖一','望远镜座','57'],[73,124.128837,9.185545,'柳宿增十','巨蟹座','31'],[74,169.835199,-14.77854,'翼宿七','巨爵座','53'],
[75,22.870873,15.345824,'右更二','双鱼座','14'],[76,309.387268,14.595092,'瓠瓜四','海豚座','69'],[77,130.898072,-33.186384,'天狗五','罗盘座','65'],[78,325.36936,-77.390045,'蛇尾三','南极座','50'],[79,126.434145,-66.136889,'飞鱼三','飞鱼座','76'],
[80,337.822424,50.282452,'螣蛇一','蝎虎座','68'],[81,48.017918,-28.986944,'天苑增三','天炉座','41'],[82,248.362848,-78.897146,'异雀八','天燕座','67'],[83,163.327938,34.214872,'势四','小狮座','64'],[84,278.801778,-8.244071,'天弁一','盾牌座','84'],
[85,63.500474,-42.294369,'天园增六','时钟座','58'],[86,115.311802,-9.551129,'参宿增二十六','麒麟座','35'],[87,318.955966,5.247845,'虚宿二','小马座','87'],[88,244.960094,-50.155506,'近波斯一','矩尺座','74'],[89,52.267232,59.940335,'传舍七','鹿豹座','18'],
[90,124.631459,-76.919721,'小斗增一','蝘蜓座','79'],[91,287.368091,-37.904473,'鳖六','南冕座','80'],[92,156.787922,-31.067777,'近天记增二','唧筒座','62'],[93,197.968307,27.878184,'周鼎一','后发座','42'],[94,14.651503,-29.357447,'近土司空南','玉夫座','36'],
[95,292.176372,24.664907,'齐增五','狐狸座','55'],[96,70.140471,-41.86375,'近天园增六','雕具座','81'],[97,151.984502,-0.371635,'天相二','六分仪座','47'],[98,315.322752,-32.257766,'璃瑜增一','显微镜座','66'],[99,82.970621,-76.340973,'山案座','山案座','75']]

def Cacalt_az(ra,dec):#输入赤经ra、赤纬dec，输出方位角AZ、俯仰角ALT
    global UT,d,LST
    HA=LST-ra
    
    sinDEC, cosDEC = math.sin(dec*math.pi/180), math.cos(dec*math.pi/180)
    sinLAT, cosLAT = math.sin(LAT*math.pi/180), math.cos(LAT*math.pi/180)
    sinHA,  cosHA  = math.sin(HA*math.pi/180),  math.cos(HA*math.pi/180) 
    
    sinALT = sinDEC*sinLAT + cosDEC*cosLAT*cosHA
    ALT = math.asin(sinALT)
    cosALT = math.cos(ALT)
    cosA   = (sinDEC-sinALT*sinLAT)/(cosALT*cosLAT)
    A = math.acos(cosA)*180/math.pi
    ALT=180*ALT/math.pi
    
    if sinHA<0:
        AZ=A
    else:
        AZ=360-A
    return AZ,ALT
    
def Cacra_dec(head,pitch):#输入方位角AZ、俯仰角ALT,输出赤经ra、赤纬dec
    global UT,d,LST
    
    sinlat, coslat = math.sin(LAT*math.pi/180), math.cos(LAT*math.pi/180)
    sina,   cosa   = math.sin(head*math.pi/180), math.cos(head*math.pi/180)
    sinalt, cosalt, tanalt = math.sin(pitch*math.pi/180), math.cos(pitch*math.pi/180),math.tan(pitch*math.pi/180)

    sindec = cosa*cosalt*coslat + sinalt*sinlat
    dec = math.asin(sindec)*180/math.pi
    t = math.atan2(-sina*cosalt,sinalt*coslat-cosalt*sinlat*cosa)*180/math.pi
    ra = (LST-t)%360
    return ra,dec

def bds_work():     #北斗解析
    global LON,LAT,UT,d,UTC
    L_bds=[]    #信息转存为列表
    Re_bds=''

    try:
        if uart1.any():
            Re_bds = str(uart1.readline(),'UTF-8')
        #Re_bds="$GNRMC,083015.000,V,2235.0736,N,11357.1265,E,,,040922,,,M*52"
        #NMEA0183，仅解析GNRMC
        if Re_bds.find("$GNRMC") < 0: 
            return 0
    
        L_bds=Re_bds.split(',') #字符串转列表
        if L_bds[0] == "$GNRMC":#已收到$GNRMC
            if L_bds[1]!='':    #存在UT
                HMS, DMY = L_bds[1], L_bds[9]   #时分秒,日月年
                dd, mm, yy, hh, ff = int(DMY[:2]), int(DMY[2:4]), int(DMY[4:]), int(HMS[:2]), int(HMS[2:4])
                UT = hh+ff/60
                UTC= "20"+str(DMY[4:])+"/"+str(DMY[2:4])+"/"+DMY[:2]+" "+HMS[:2]+":"+HMS[2:4]
                d  = 8034.5+(yy-22)*365+days[mm-1]+dd+UT/24
    
                if L_bds[3]!='':
                    N_lat, E_lon  =L_bds[3], L_bds[5]  #获得经纬度原始数据
                    LAT,LON=bds_location(N_lat,E_lon)  #解析经纬度转为度的浮点数            
            return 1
    except:
        return 0

def bds_location(n,e):              #经纬度解析
    n_dos, e_dos = n.find("."), e.find(".")
    dd_n,  mm_n  = n[:n_dos-2], n[n_dos-2:] #纬度的度分
    dd_e,  mm_e  = e[:e_dos-2], e[e_dos-2:] #经度的度分
    dn=float(dd_n)+float(mm_n)/60   #纬度：dddmm.mmmm转为度
    de=float(dd_e)+float(mm_e)/60   #经度：ddmm.mmmm转为度
    return dn,de
    
def starscan_show(ra,dec):               #遍历星表，星星匹配附近正负5
    for i in range(len(STAR)):
        if (abs(STAR[i][1]-ra)<5 or abs(STAR[i][1]-ra) >355) and abs(STAR[i][2]-dec)<5:
            oled.DispChar(str(STAR[i][3]), 104-6*len(STAR[i][3]), 50,1)
        if (abs(STAR[i][1]-ra)<10 or abs(STAR[i][1]-ra) >350) and abs(STAR[i][2]-dec)<10:
            oled.blit(image_picture.load(STAR[i][5]+'.pbm', 1), 80, -5)
            oled.DispChar(str(STAR[i][4]), 104-6*len(STAR[i][4]), 35,1)
            break

def on_button_a_pressed(_):         #A键判断是否进行磁力校准
    global calibrate_start
    calibrate_start= not calibrate_start
        
def test(_):                        #每1秒测试1次
    for i in range(10):             
        if bds_work() == 1: break
        
def magnetic_calibrate():           #磁力校准
    global Xmax,Xmin,Ymax,Ymin,Zmax,Zmin
    mx,my,mz = magnetic.get_x(),magnetic.get_y(),magnetic.get_z()
    Xmax,Xmin,Ymax,Ymin,Zmax,Zmin = max(Xmax,mx),min(Xmin,mx),max(Ymax,my),min(Ymin,my),max(Zmax,mz),min(Zmin,mz)

def head_tilt():                    #倾斜补偿
    global Xmax,Xmin,Ymax,Ymin,Zmax,Zmin
   
    roll,pitch = accelerometer.roll_pitch_angle()
    sinRoll,cosRoll = math.sin(roll/180*math.pi),math.cos(roll/180*math.pi)
    sinPitch,cosPitch = math.sin(pitch/180*math.pi),math.cos(pitch/180*math.pi)
    
    mx,my,mz = magnetic.get_x(),magnetic.get_y(),magnetic.get_z()
    MX=mx-(Xmax + Xmin)/2
    MY=(my-(Ymax + Ymin)/2)*(Xmax-Xmin)/(Ymax-Ymin)
    MZ=(mz-(Zmax + Zmin)/2)*(Xmax-Xmin)/(Zmax-Zmin) 
    
    xh = MX*cosPitch + MY*sinRoll*sinPitch + MZ*cosRoll*sinPitch
    yh = -MZ*sinRoll + MY*cosRoll      
    bearing = (math.atan2(xh,yh)*180/math.pi-90-3.3)%360    #深圳，磁偏角-318′

    return bearing

while True:
    if init==0:
        image_picture = Image()
    
        calibrate_start=False #是否进行校准
        laser_on=False
        uart1 = UART(1, baudrate=9600, tx=Pin.P16, rx=Pin.P15)      #北斗接口
        tim1.init(period=1000, mode=Timer.PERIODIC,callback=test)   #每1秒测试1次
        button_a.event_pressed = on_button_a_pressed
        init=1

    if calibrate_start==True:
        magnetic_calibrate()
        oled.blit(image_picture.load('calibrate.pbm', 0), 0, 0)
        oled.show()
        continue 
    
    LST=(100.46 + 0.985647 *d + LON + 15*UT)%360
    
    A = head_tilt()                         #有倾斜补偿的真方位角
    H = accelerometer.roll_pitch_angle()[1] #俯仰角（高度角）
    
    ra,dec = Cacra_dec(A,H)                 #计算赤经赤纬
    #a,h=Cacalt_az(ra,dec)
    
    oled.fill(0)
    starscan_show(ra,dec)               #遍历星表,显示星星和星座
    oled.DispChar(UTC[5:], 0, 0, 1)
    oled.DispChar("N:{:.5f} ".format(LAT), 0, 16, 1)
    oled.DispChar("E:{:.5f} ".format(LON), 0, 32, 1) 
    oled.DispChar('A:'+str(int(A))+' H:'+str(int(H)), 0, 48,1)
    #oled.DispChar('RA:'+str(int(ra))+' DEC:'+str(int(dec)),0,32,1)
    oled.show()
    sleep_ms(1000) 
