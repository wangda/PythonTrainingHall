#-*- coding: utf-8 -*-

import wx
import win32api
import sys, os, time
import threading

APP_TITLE = u'定时器和线程'
APP_ICON = 'res/python.ico'

class mainFrame(wx.Frame):
    '''程序主窗口类，继承自wx.Frame'''
    
    def __init__(self, parent):
        '''构造函数'''
        
        wx.Frame.__init__(self, parent, -1, APP_TITLE)
        self.SetBackgroundColour(wx.Colour(224, 224, 224))
        self.SetSize((320, 300))
        self.Center()
        
        if hasattr(sys, "frozen") and getattr(sys, "frozen") == "windows_exe":
            exeName = win32api.GetModuleFileName(win32api.GetModuleHandle(None))
            icon = wx.Icon(exeName, wx.BITMAP_TYPE_ICO)
        else :
            icon = wx.Icon(APP_ICON, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)
        
        #font = wx.Font(24, wx.DECORATIVE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, 'Comic Sans MS')
        font = wx.Font(30, wx.DECORATIVE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, 'Monaco')
        
        self.clock = wx.StaticText(self, -1, u'08:00:00', pos=(50,50), size=(200,50), style=wx.TE_CENTER|wx.SUNKEN_BORDER)
        self.clock.SetForegroundColour(wx.Colour(0, 224, 32))
        self.clock.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.clock.SetFont(font)
        
        self.stopwatch = wx.StaticText(self, -1, u'0:00:00.0', pos=(50,150), size=(200,50), style=wx.TE_CENTER|wx.SUNKEN_BORDER)
        self.stopwatch.SetForegroundColour(wx.Colour(0, 224, 32))
        self.stopwatch.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.stopwatch.SetFont(font)
        
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)
        self.timer.Start(50)
        
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        
        self.sec_last = None
        self.is_start = False
        self.t_start = None
        
        thread_sw = threading.Thread(target=self.StopWatchThread)
        thread_sw.setDaemon(True)
        thread_sw.start()
        
    def OnTimer(self, evt):
        '''定时器函数'''
        
        t = time.localtime()
        if t.tm_sec != self.sec_last:
            self.clock.SetLabel('%02d:%02d:%02d'%(t.tm_hour, t.tm_min, t.tm_sec))
            self.sec_last = t.tm_sec
        
    def OnKeyDown(self, evt):
        '''键盘事件函数'''
        
        if evt.GetKeyCode() == wx.WXK_SPACE:
            self.is_start = not self.is_start
            self.t_start= time.time()
        elif evt.GetKeyCode() == wx.WXK_ESCAPE:
            self.is_start = False
            self.stopwatch.SetLabel('0:00:00.0')
        
    def StopWatchThread(self):
        '''线程函数'''
        
        while True:
            if self.is_start:
                n = int(10*(time.time() - self.t_start))
                deci = n%10
                ss = int(n/10)%60
                mm = int(n/600)%60
                hh = int(n/36000)
                wx.CallAfter(self.stopwatch.SetLabel, '%d:%02d:%02d.%d'%(hh, mm, ss, deci))
            time.sleep(0.02)
        
class mainApp(wx.App):
    def OnInit(self):
        self.SetAppName(APP_TITLE)
        self.Frame = mainFrame(None)
        self.Frame.Show()
        return True

if __name__ == "__main__":
    app = mainApp()
    app.MainLoop()
