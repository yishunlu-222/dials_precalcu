from __future__ import annotations

import matplotlib.pyplot as plt
import zmq

if __name__ == "__main__":

    port = "tcp://127.0.0.1:5555"

    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(port)

    filter_ = "dials.ssx_index"
    socket.setsockopt_string(zmq.SUBSCRIBE, filter_)

    import numpy as np

    x = np.array([])
    y = np.array([])

    plt.ion()  # <-- work in "interactive mode"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Image number")
    ax.set_ylabel(f"{'%'} of strong spots indexed")
    # ax.canvas.set_window_title('Live Chart')
    # ax.set_title("Indexing hits")
    import json

    while True:

        event_count = socket.poll(timeout=1000)
        if event_count:
            new_x = []
            new_y = []
            for _ in range(event_count):
                message = socket.recv_string()
                print("Received ZMQ Message: ", message)
                message = json.loads(message.lstrip("dials.ssx_index"))
                img = int(message["image_no"])
                n_spots = int(message["n_indexed"])
                n_strong = int(message["n_strong"])
                pc_indexed = 100.0 * n_spots / n_strong
                new_x.append(img)
                new_y.append(pc_indexed)
            x = np.concatenate([x, np.array(new_x)])
            y = np.concatenate([y, np.array(new_y)])

            ax.clear()
            ax.set_ylim([0, 100])
            ax.set_xlabel("Image number")
            ax.set_ylabel(f"{'%'} of strong spots indexed")
            ax.scatter(x=x, y=y, color="r")

        # show the plot
        plt.show()
        plt.pause(0.0001)
        # else:
        #    print("no messages")
