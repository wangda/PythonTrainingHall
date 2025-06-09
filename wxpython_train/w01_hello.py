import wx

#创建应用程序对象
app=wx.App()
#创建顶级窗口对象
frame=wx.Frame(None,title='wxPython',size=(500,600))
#显示窗口
frame.Show()
#调用应用程序的主事件循环方法
app.MainLoop()