import matplotlib.pyplot as plt


def plot_room_views(room):
    # Create a new figure
    fig = plt.figure(figsize=(15, 6))

    # Create subplot for the top view
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    room.plot(fig=fig, ax=ax1)
    ax1.view_init(90, -90)
    ax1.set_title("Top View")

    # Create subplot for the front view
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    room.plot(fig=fig, ax=ax2)
    ax2.view_init(0, -90)
    ax2.set_title("Front View")

    # Create subplot for the side view
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    room.plot(fig=fig, ax=ax3)
    ax3.view_init(0, 0)
    ax3.set_title("Side View")

    # Show the plot
    plt.show()
