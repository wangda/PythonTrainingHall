import wx

class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        super().__init__(parent=parent, id=id, title="wxPython标题", size=(300, 180))
        panel = wx.Panel(parent=self)
        title = wx.StaticText(parent=panel, label="回乡偶书二首 -- 其一", pos=(100, 20))
        font = wx.Font(pointSize=16, family=wx.FONTFAMILY_DEFAULT, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_BOLD)
        title.SetFont(font=font)
        wx.StaticText(parent=panel, label="少小离家老大回", pos=(100, 40))
        wx.StaticText(parent=panel, label="乡音未改鬓毛衰", pos=(100, 60))
        wx.StaticText(parent=panel, label="儿童相见不相识", pos=(100, 80))
        wx.StaticText(parent=panel, label="笑问客从何处来", pos=(100, 100))

        self.Center()

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame(parent=None, id=-1)
    frame.Show()
    app.MainLoop()

