from math import pi, cos, sin
from sys import argv
from typing import List, Tuple

from OpenGL.GL import *
from OpenGL.GLUT import *

DEG_TO_RAD = pi / 180
MENU_ITEM_WIDTH = 170

ww, wh = 1024, 768
ncon = None

funcs = []
currentFunc = 0
prevFunc = currentFunc


def drawMenus():
    # draw horizonal and vertical split lines
    glColor3f(0, 0, 0)
    glLineWidth(1)

    glBegin(GL_LINES)
    for i in range(len(funcs) + 1):
        glVertex2f(-ww / 2, wh / 2 - (50 * i))
        glVertex2f(-ww / 2 + MENU_ITEM_WIDTH, wh / 2 - (50 * i))

    glVertex2f(-ww / 2, +wh / 2)
    glVertex2f(+ww / 2, +wh / 2)
    glVertex2f(-ww / 2 + MENU_ITEM_WIDTH, -wh / 2)
    glVertex2f(-ww / 2 + MENU_ITEM_WIDTH, +wh / 2)
    glEnd()

    # highlight current function
    glColor3f(0, 1, 0)
    glBegin(GL_QUADS)
    glVertex2f(-ww / 2, wh / 2 - (currentFunc * 50) - 1)
    glVertex2f(-ww / 2 + MENU_ITEM_WIDTH - 1, wh / 2 - (currentFunc * 50) - 1)
    glVertex2f(-ww / 2 + MENU_ITEM_WIDTH - 1, wh / 2 - (currentFunc * 50) - 50)
    glVertex2f(-ww / 2, wh / 2 - (currentFunc * 50) - 50)
    glEnd()

    # add text to each "buttons"
    glColor3f(0, 0, 0)
    for i in range(len(funcs)):
        glRasterPos2f(-ww / 2 + 7, wh / 2 - (50 * i) - 17)
        for ch in bytearray(chr(ord(b'0') + i + 1), encoding='ascii'):
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ch)

        glRasterPos2f(-ww / 2 + 7, wh / 2 - (50 * i) - 43)
        for ch in funcs[i][0]:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ch)

    # display docstring for current function
    glColor3f(0, 0, 0)
    y = wh / 2 - 5
    for line in funcs[currentFunc][3].splitlines():
        glRasterPos2f(-ww / 2 + MENU_ITEM_WIDTH + 5, y)
        for ch in bytearray(line, encoding='ascii'):
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ch)
        y -= 15


def myReshape(w, h):
    global ww, wh
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-w / 2, w / 2, -h / 2, h / 2, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    ww, wh = w, h


def mouseToWindowCoord(x, y):
    x = x - (ww / 2)
    y = (wh / 2) - y
    return x, y


def nearestPoint(wx, wy, points, tolerance=400):
    minDistance2 = tolerance + 114514
    minID = -1
    for i in range(len(points)):
        distance2 = (points[i][0] - wx) ** 2 + (points[i][1] - wy) ** 2
        if distance2 < minDistance2:
            minDistance2 = distance2
            minID = i
    if minDistance2 <= tolerance:
        return minID
    return None


def dotProduct(a, b):
    m, n, p = len(a), len(b), len(b[0])
    res = []
    for i in range(m):
        res.append([])
        for j in range(p):
            res[i].append(sum([a[i][r] * b[r][j] for r in range(n)]))
    return res


######################################################################################### BEIZER

docBeizer = '''
Left click on empty place to add a control point.
Left drag a control point to move it.
Right click a control point to delete it.
'''

controls = []


def beizerMotion(mx, my):
    global controls, ncon
    wx, wy = mouseToWindowCoord(mx, my)
    controls[ncon] = (wx, wy)


def beizerMouse(button, state, wx, wy):
    global ncon, controls
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        glutMotionFunc(beizerMotion)
        ncon = nearestPoint(wx, wy, controls)
        if ncon is None:
            controls.append((wx, wy))
            ncon = len(controls) - 1
    elif button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
        ncon = nearestPoint(wx, wy, controls)
        if ncon is not None:
            controls.pop(ncon)
    else:
        ncon = None
        glutMotionFunc(None)


def nCr(n):
    facts = [1]
    for i in range(n):
        facts.append(facts[-1] * (i + 1))
    ans = []
    for r in range(n + 1):
        ans.append(facts[n] // (facts[r] * facts[n - r]))
    return ans


def calcBezier(control: List[Tuple[float, float]], interval=0.01):
    n = len(control) - 1
    xCon = nCr(n)
    yCon = xCon.copy()
    res = []
    for i in range(n + 1):
        xCon[i] *= control[i][0]
        yCon[i] *= control[i][1]

    t = 0.
    while t <= 1 + interval * 0.95:
        if t > 1:
            t = 1
        res.append((
            sum([xCon[i] * (t ** i) * ((1 - t) ** (n - i)) for i in range(n + 1)]),
            sum([yCon[i] * (t ** i) * ((1 - t) ** (n - i)) for i in range(n + 1)]),
        ))
        t += interval

    return res


def clearCtrlPoints():
    global controls, currentFunc
    controls = []
    currentFunc = prevFunc


def drawBeizer():
    global controls
    if (not controls):
        return
    points = calcBezier(controls)

    glPointSize(20)
    glColor3f(0, 0, 1)
    glBegin(GL_POINTS)
    for point in controls:
        glVertex2f(point[0], point[1])
    glEnd()

    glLineWidth(3)
    glColor3f(0, 1, 0)
    glBegin(GL_LINE_STRIP)
    for point in controls:
        glVertex2f(point[0], point[1])
    glEnd()

    glPointSize(4)
    glColor3f(1, 0, 0)
    glBegin(GL_LINE_STRIP)
    for point in points:
        glVertex2f(point[0], point[1])
    glEnd()


######################################################################################### B-SPINE

order = 3
docBSpline = docBeizer + f"\n Open Uniform B-Spline Curve, Order: {order}"


def drawBSpline(uInterval=0.01):
    global controls, order
    n = len(controls) - 1
    m = order
    # n 为控制顶点个数-1，controls[k] (k = 0..n) 即是 n+1 个控制顶点
    # order 为阶参数，也就是 m，重命名只是因为全局变量用单字母容易混淆

    if m == n + 1:
        # 当 m = n+1 时开放 B-Spline 就是 Beizer Curve
        return drawBeizer()

    L, k = n - m, n - m + 2
    t = [0] * m
    t.extend(range(1, k))
    t.extend([k] * m)
    # ti = [(m 个 0), (1..k-1), (m 个 k)]

    points = []
    # 降阶过程，参考 https://zhuanlan.zhihu.com/p/50450278
    for j in range(m - 1, n + 1):
        uList = [t[j] + uInterval * uid for uid in range(int((t[j + 1] - t[j]) / uInterval))] + [t[j + 1]]
        for u in uList:
            progCtrls = controls.copy()
            for r in range(1, order):
                for i in range(j, j - order + r, -1):
                    x1, x2, y1, y2 = u - t[i], t[i + order - r] - t[i], t[i + order - r] - u, t[i + order - r] - t[i]

                    coef1 = 0 if x1 == x2 == 0 else x1 / x2
                    coef2 = 0 if y1 == y2 == 0 else y1 / y2
                    progCtrls[i] = ((progCtrls[i][0] * coef1 + progCtrls[i - 1][0] * coef2),
                                    (progCtrls[i][1] * coef1 + progCtrls[i - 1][1] * coef2))
            points.append(progCtrls[j])

    glPointSize(20)
    glColor3f(0, 0, 1)
    glBegin(GL_POINTS)
    for point in controls:
        glVertex2f(point[0], point[1])
    glEnd()

    glLineWidth(3)
    glColor3f(0, 1, 0)
    glBegin(GL_LINE_STRIP)
    for point in controls:
        glVertex2f(point[0], point[1])
    glEnd()

    glPointSize(4)
    glColor3f(1, 0, 0)
    glBegin(GL_LINE_STRIP)
    for point in points:
        glVertex2f(point[0], point[1])
    glEnd()


def bspOrderChange(delta=1):
    global order, currentFunc, prevFunc, docBSpline
    order += delta
    if order < 2:
        order = 2
    docBSpline = docBeizer + f"\n Open Uniform B-Spline Curve, Order: {order}"
    for i in range(len(funcs)):
        if funcs[i][1] == drawBSpline:
            funcs[i][3] = docBSpline
            currentFunc = i
            return


######################################################################################### BRESENHAM LINE/CIRCLE

docBresenham = '''
Drag a control point to move it.
Left drag on empty place to rotate.
Right drag on empty place to move.

The starting point of drag is the
reference point of rotation,
drag left to rotate clockwise, 
right to rotate counterclockwise.
'''
endpoints = [(-50, -50), (150, 50)]
frozenEndpoints = []
rotCenter = (-114.514, -1919.810)
rotDegree = 0
sizeMul = 1
movX, movY = 0, 0
rot, mov = 0, 0


def generateRotateMatrix(centerX, centerY, thetaInDegree, sizeMul=1):
    T1 = [[1, 0, +centerX],
          [0, 1, +centerY],
          [0, 0, 1]]
    cosTheta = cos(thetaInDegree * DEG_TO_RAD)
    sinTheta = sin(thetaInDegree * DEG_TO_RAD)
    T2 = [[+cosTheta, -sinTheta, 0],
          [+sinTheta, +cosTheta, 0],
          [0, 0, 1]]
    T3 = [[sizeMul, 0, 0],
          [0, sizeMul, 0],
          [0, 0, 1]]
    T4 = [[1, 0, -centerX],
          [0, 1, -centerY],
          [0, 0, 1]]
    return dotProduct(dotProduct(dotProduct(T1, T2), T3), T4)


def rotatePoints(points, rotCenter, rotDegree, sizeMul=1):
    origMatrix = [[],
                  [],
                  []]
    for point in points:
        origMatrix[0].append(point[0])
        origMatrix[1].append(point[1])
        origMatrix[2].append(1)

    rotateMatrix = generateRotateMatrix(rotCenter[0], rotCenter[1], rotDegree, sizeMul)

    resMatrix = dotProduct(rotateMatrix, origMatrix)
    resPoints = []
    for i in range(len(points)):
        resPoints.append((resMatrix[0][i], resMatrix[1][i]))
    return resPoints


def movePoints(points, deltaX, deltaY):
    origMatrix = [[],
                  [],
                  []]
    for point in points:
        origMatrix[0].append(point[0])
        origMatrix[1].append(point[1])
        origMatrix[2].append(1)

    moveMatrix = [[1, 0, deltaX],
                  [0, 1, deltaY],
                  [0, 0, 1]]

    resMatrix = dotProduct(moveMatrix, origMatrix)
    resPoints = []
    for i in range(len(points)):
        resPoints.append((resMatrix[0][i], resMatrix[1][i]))
    return resPoints


def bresMotion(mx, my):
    global controls, ncon, endpoints, rotDegree, sizeMul, movX, movY
    wx, wy = mouseToWindowCoord(mx, my)
    if ncon is not None:
        endpoints[ncon] = (wx, wy)
    else:
        if rot:
            rotDegree = wx - rotCenter[0]
            sizeMul = 1 + (wy - rotCenter[1]) / 50
            endpoints = rotatePoints(frozenEndpoints, rotCenter, rotDegree, sizeMul)
        if mov:
            movX, movY = wx - rotCenter[0], wy - rotCenter[1]
            endpoints = movePoints(frozenEndpoints, movX, movY)


def bresMouse(button, state, wx, wy):
    global ncon, endpoints, frozenEndpoints, rotCenter, rotDegree, sizeMul, rot, mov, movX, movY
    if state == GLUT_DOWN:
        ncon = nearestPoint(wx, wy, endpoints)
        frozenEndpoints = endpoints.copy()
        glutMotionFunc(bresMotion)
        if ncon is None:
            rotCenter = (wx, wy)
            rotDegree, sizeMul, movX, movY = 0, 1, 0, 0
            if button == GLUT_LEFT_BUTTON:
                rot, mov = 1, 0
            elif button == GLUT_RIGHT_BUTTON:
                rot, mov = 0, 1
        else:
            rot, mov = 0, 0
    else:
        ncon = None
        if rot == 1:
            endpoints = rotatePoints(frozenEndpoints, rotCenter, rotDegree, sizeMul)
        if mov == 1:
            endpoints = movePoints(frozenEndpoints, movX, movY)
        rot, mov = 0, 0
        glutMotionFunc(None)


def vertex2fCircle8(x, y, centerX, centerY):
    for sgnX in (1, -1):
        for sgnY in (1, -1):
            for (dx, dy) in [(x, y), (y, x)]:
                glVertex2f(centerX + (sgnX * dx), centerY + (sgnY * dy))


def drawBresCircle(centerX, centerY, radius):
    cx, cy = 0, radius
    d = 3 - 2 * radius
    glBegin(GL_POINTS)
    while cx <= cy:
        vertex2fCircle8(cx, cy, centerX, centerY)
        if d < 0:
            d = d + 4 * cx + 6
        else:
            d = d + 4 * (cx - cy) + 10
            cy -= 1
        cx += 1
    glEnd()


def drawBresLine(x1, y1, x2, y2):
    x1, y1, x2, y2 = map(lambda x: int(x + .5), (x1, y1, x2, y2))
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    xySwap = 0

    if dx < dy:
        xySwap = 1
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx

    ix = 1 if (x2 - x1 > 0) else -1
    iy = 1 if (y2 - y1 > 0) else -1
    cx, cy = x1, y1

    n2dy = dy * 2
    n2dydx = (dy - dx) * 2
    d = dy * 2 - dx

    glPointSize(4)
    glColor3f(1, 0, 0)
    glBegin(GL_POINTS)

    while cx != x2:
        if d < 0:
            d += n2dy
        else:
            cy += iy
            d += n2dydx
        if xySwap:
            glVertex2f(cy, cx)
        else:
            glVertex2f(cx, cy)
        cx += ix
    glEnd()


def drawMovChanges():
    if rot or mov:
        glPointSize(10)
        glColor3f(.5, .5, 0)
        glBegin(GL_POINTS)
        glVertex2f(rotCenter[0], rotCenter[1])
        glEnd()

    if rot:
        glColor3f(0, 0, 0)
        glRasterPos2f(rotCenter[0] - 62, rotCenter[1] + 3)
        rotStr = f"Rotate {rotDegree:5.0f} deg"
        for ch in bytearray(rotStr, encoding='ascii'):
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ch)
        glRasterPos2f(rotCenter[0] - 62, rotCenter[1] -12)
        rr = f"Resize {sizeMul:5.2f} x"
        for ch in bytearray(rr, encoding='ascii'):
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ch)
    if mov:
        glColor3f(0, 0, 0)
        glRasterPos2f(rotCenter[0] - 44, rotCenter[1] - 3)
        rr = f"Move"
        for ch in bytearray(rr, encoding='ascii'):
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ch)
        glRasterPos2f(rotCenter[0] + 10, rotCenter[1] + 3)
        rr = f"dX {movX:4.0f}"
        for ch in bytearray(rr, encoding='ascii'):
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ch)
        glRasterPos2f(rotCenter[0] + 10, rotCenter[1] - 12)
        rr = f"dY {movY:4.0f}"
        for ch in bytearray(rr, encoding='ascii'):
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ch)


def drawBresenham():
    # draw control points
    glPointSize(20)
    glBegin(GL_POINTS)
    for point in endpoints:
        glColor3f(0, 0, 1)
        glVertex2f(point[0], point[1])
    glEnd()

    x1, y1, x2, y2 = int(endpoints[0][0]), int(endpoints[0][1]), int(endpoints[1][0]), int(endpoints[1][1])
    drawBresLine(x1, y1, x2, y2)

    dx, dy = abs(x2 - x1), abs(y2 - y1)
    radius = int((dx ** 2 + dy ** 2) ** 0.5 + 0.5)
    drawBresCircle(x1, y1, radius)

    drawMovChanges()


######################################################################################### LIANG-BARSKY

docLiang = '''
Drag a control point to move it.
'''

visibleRegion = [(-100, 100), (170, -100)]


def liangMotion(mx, my):
    global controls, ncon, visibleRegion
    wx, wy = mouseToWindowCoord(mx, my)
    if ncon is not None:
        endpoints[ncon] = (wx, wy)
        if len(endpoints) == 4:
            visibleRegion = endpoints[2:]


def liangMouse(button, state, wx, wy):
    global ncon, endpoints, visibleRegion
    if state == GLUT_DOWN:
        if len(endpoints) == 2:
            endpoints = endpoints + visibleRegion
        ncon = nearestPoint(wx, wy, endpoints)
        glutMotionFunc(liangMotion)
    else:
        ncon = None
        if len(endpoints) == 4:
            endpoints, visibleRegion = endpoints[:2], endpoints[2:]
        glutMotionFunc(None)


def drawLiang():
    # draw control points
    glLineWidth(3)
    glBegin(GL_QUADS)
    glColor3f(1, .7, 0)
    glVertex2f(visibleRegion[0][0], visibleRegion[0][1])
    glVertex2f(visibleRegion[0][0], visibleRegion[1][1])
    glVertex2f(visibleRegion[1][0], visibleRegion[1][1])
    glVertex2f(visibleRegion[1][0], visibleRegion[0][1])
    glEnd()
    glPointSize(20)
    glBegin(GL_POINTS)
    for point in visibleRegion:
        glColor3f(0, 0, 1)
        glVertex2f(point[0], point[1])
    for point in endpoints:
        glColor3f(0, 0, 1)
        glVertex2f(point[0], point[1])
    glEnd()

    x1, y1, x2, y2 = endpoints[0][0], endpoints[0][1], endpoints[1][0], endpoints[1][1]

    glLineWidth(2)
    glColor3f(0, 1, 0)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()

    xl, xr, yu, yd = visibleRegion[0][0], visibleRegion[1][0], visibleRegion[0][1], visibleRegion[1][1]

    if xl > xr:
        xl, xr = xr, xl
    if yu < yd:
        yu, yd = yd, yu

    p = [x1 - x2, x2 - x1, y1 - y2, y2 - y1]
    if not any(p):
        return

    q = [x1 - xl, xr - x1, y1 - yd, yu - y1]
    if p[0] == 0:
        if q[0] < 0 or q[1] < 0:
            return
        kmx, kmn = [k for k in [2, 3] if p[k] < 0], [k for k in [2, 3] if p[k] > 0]
    elif p[2] == 0:
        if q[2] < 0 or q[3] < 0:
            return
        kmx, kmn = [k for k in [0, 1] if p[k] < 0], [k for k in [0, 1] if p[k] > 0]
    else:
        kmx, kmn = [k for k in range(4) if p[k] < 0], [k for k in range(4) if p[k] > 0]

    umx = max([q[k] / p[k] for k in kmx] + [0])
    umn = min([q[k] / p[k] for k in kmn] + [1])
    if umx > umn:
        return
    dx, dy = x2 - x1, y2 - y1
    xs, xt, ys, yt = x1 + umx * dx, x1 + umn * dx, y1 + umx * dy, y1 + umn * dy
    drawBresLine(xs, ys, xt, yt)


######################################################################################### MAIN INITS

def TBD(a=None, b=None, c=None, d=None):
    pass


funcs.append([b"BeizerCurve", drawBeizer, beizerMouse, docBeizer])
funcs.append([b"B-Spline", drawBSpline, beizerMouse, docBSpline])
funcs.append([b"  Order +1", bspOrderChange, TBD, docBeizer])
funcs.append([b"  Order -1", lambda: bspOrderChange(-1), TBD, docBeizer])
funcs.append([b"Clear Ctrl Points", clearCtrlPoints, TBD, docBeizer])
funcs.append([b"Bresenham & Rotate", drawBresenham, bresMouse, docBresenham])
funcs.append([b"Liang-Barsky", drawLiang, liangMouse, docLiang])

menu = 0


def processMenu(opt):
    global currentFunc, prevFunc
    prevFunc, currentFunc = currentFunc, opt
    return 0


def setMenu():
    global menu
    menu = glutCreateMenu(processMenu)
    for i in range(len(funcs)):
        glutAddMenuEntry(funcs[i][0], i)
    glutAttachMenu(GLUT_MIDDLE_BUTTON)


def myKeyboard(key, x, y):
    global currentFunc, prevFunc
    try:
        key = int(key) - 1
        if key in range(len(funcs)):
            currentFunc, prevFunc = key, currentFunc
    except:
        return


def myMouse(button, state, mx, my):
    global currentFunc, prevFunc
    wx, wy = mouseToWindowCoord(mx, my)
    if mx <= MENU_ITEM_WIDTH and state == GLUT_DOWN:
        selectedFunc = (my // 50)
        if selectedFunc < len(funcs):
            currentFunc, prevFunc = selectedFunc, currentFunc
    else:
        funcs[currentFunc][2](button, state, wx, wy)


def myDisplay():
    glClear(GL_COLOR_BUFFER_BIT)
    drawMenus()
    funcs[currentFunc][1]()
    glutSwapBuffers()


def idle():
    glutPostRedisplay()


def init(*args):
    glutInit(*args)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(ww, wh)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"CG-MA")
    glClearColor(1, 1, 1, 1)
    glutDisplayFunc(myDisplay)
    glutReshapeFunc(myReshape)
    glutMouseFunc(myMouse)
    glutKeyboardFunc(myKeyboard)
    glutIdleFunc(idle)
    setMenu()

    glutMainLoop()


if __name__ == '__main__':
    init(*argv)
