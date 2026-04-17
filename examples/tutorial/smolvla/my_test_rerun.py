import rerun as rr
import numpy as np
import cv2

rr.init("camera_demo", spawn=True)

def show_video():
    for i in range(100):
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        # rr.set_time_sequence("frame", i)
        rr.set_time("frame", sequence=i)
        rr.log("camera/image", rr.Image(img))

        attn = np.random.rand(16, 16)

        # resize 到图像大小
        attn_resized = cv2.resize(attn, (320, 240))

        # 转 heatmap
        attn_color = cv2.applyColorMap(
            (attn_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        rr.log("camera/attention", rr.Image(attn_color))

        # overlay camera and heatmap
        overlay = cv2.addWeighted(img, 0.6, attn_color, 0.4, 0)

        rr.log("camera/overlay", rr.Image(overlay))

        show_action()



def show_heatmap():
    attn = np.random.rand(16, 16)

    # resize 到图像大小
    attn_resized = cv2.resize(attn, (320, 240))

    # 转 heatmap
    attn_color = cv2.applyColorMap(
        (attn_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    rr.log("camera/attention", rr.Image(attn_color))


def show_action():
    action = np.random.randn(7)

    rr.log("policy/action", rr.Tensor(action))


if __name__ == "__main__":
    show_video()