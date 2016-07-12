import wx

app = wx.App(False)

frame = wx.Frame(None, title="Draw on Panel")
panel = wx.Panel(frame)

def on_paint(event):
    dc = wx.PaintDC(event.GetEventObject())
    dc.Clear()
    dc.SetPen(wx.Pen("red", 2))
    def hyp(c):
      def f(x):
        return c/x
      return f
    def p(f, x):
      zoom = 100
      d = 400
      try:
        return f((x - d) / zoom) * zoom + d
      except:
        return 0
    for x in range(0, 8000):
      dc.DrawPoint(x/10, p(hyp(9), x/10))
      dc.DrawPoint(x/10, p(hyp(3), x/10))
      dc.DrawPoint(x/10, p(hyp(1), x/10))
      dc.DrawPoint(x/10, p(hyp(0.333), x/10))
      dc.DrawPoint(x/10, p(hyp(0.111), x/10))
    dc.SetPen(wx.Pen("blue", 1))

    points = [100, 300, 366.667]
    pointlist = [(x, p(hyp(1), x)) for x in points]
#    pointlist[1] = (367, 367)
#    pointlist[1] = (340, 340)
    print pointlist


    dc.SetPen(wx.Pen("red", 5))
    for p in pointlist:
      dc.DrawCircle(p[0], p[1], 2)
    gc = wx.GraphicsContext.Create(dc)
    if gc:
      gc.SetPen(wx.BLACK_PEN)
      path = gc.CreatePath()
      curve = gc.CreatePath()
      p = 0
      path.MoveToPoint(pointlist[p][0], pointlist[p][1])
      curve.MoveToPoint(pointlist[p][0], pointlist[p][1])
      while p < (len(pointlist) - 1):
        path.AddQuadCurveToPoint(pointlist[p + 1][0], pointlist[p + 1][1], \
                                 pointlist[p + 2][0], pointlist[p + 2][1])
        curve.AddArcToPoint(pointlist[p + 1][0], pointlist[p + 1][1], pointlist[p + 2][0], pointlist[p + 2][1], 1)
        p = p + 2
      gc.StrokePath(path)
      gc.SetPen(wx.BLUE_PEN)
      gc.StrokePath(curve)

panel.Bind(wx.EVT_PAINT, on_paint)

frame.SetSize((800,800))
frame.Show(True)
app.MainLoop()
