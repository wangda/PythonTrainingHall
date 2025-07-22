import wx

class MyApp(wx.App):
    def OnInit(self):
        self.frame = wx.Frame(parent=None, title="wxPython标题", style=wx.DEFAULT_FRAME_STYLE, size=(800, 600))
        self.frame.Center()
        self.frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()