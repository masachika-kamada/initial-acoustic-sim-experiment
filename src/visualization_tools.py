import matplotlib.pyplot as plt


def plot_room(room):
    room_dim = room.get_bbox()[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    room.plot(ax=ax)

    # プロット範囲を部屋の大きさに合わせる
    ax.set_xlim([0, room_dim[0]])
    ax.set_ylim([0, room_dim[1]])
    ax.set_zlim([0, room_dim[2]])
    ax.set_box_aspect(room_dim)
    plt.show()


def plot_room_views(room, zoom_center=None, zoom_size=None):
    # Get the room dimensions from the bounding box
    room_dim = room.get_bbox()[:, 1]

    fig = plt.figure(figsize=(15, 6))
    views = [(90, -90, "Top View"), (0, -90, "Front View"), (0, 0, "Side View")]

    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        room.plot(fig=fig, ax=ax)
        ax.view_init(elev, azim)
        ax.set_title(title)
        ax.set_xlim([0, room_dim[0]])
        ax.set_ylim([0, room_dim[1]])
        ax.set_zlim([0, room_dim[2]])

        if zoom_center is not None and zoom_size is not None:
            ax.set_xlim([zoom_center[0] - zoom_size / 2, zoom_center[0] + zoom_size / 2])
            ax.set_ylim([zoom_center[1] - zoom_size / 2, zoom_center[1] + zoom_size / 2])
            ax.set_zlim([zoom_center[2] - zoom_size / 2, zoom_center[2] + zoom_size / 2])
        else:
            ax.set_xlim([0, room_dim[0]])
            ax.set_ylim([0, room_dim[1]])
            ax.set_zlim([0, room_dim[2]])

        ax.set_box_aspect(room_dim)
    plt.show()
