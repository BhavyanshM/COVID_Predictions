from matplotlib.pylab import *

plt.style.use('seaborn-pastel')

n = 1000
a_mean = np.array([3.9, -10.5])
a_cov = np.array([[1, 0], [0, 1]]) * 0.1
a = np.random.multivariate_normal(a_mean, a_cov, (n,)).T

b_mean = np.array([3.9, 8.8])
b_cov = np.array([[1, 0], [0, 1]]) * 0.1
b = np.random.multivariate_normal(b_mean, b_cov, (n,)).T

c = np.linspace(-4,6, n)
d = np.square(c)
cd = np.vstack([c,d])

a += cd
b += cd

classes = np.array([-1, 1])
labels_a = -1 * np.ones((1, n))[0, :]
labels_b = np.ones((1, n))[0, :]
labels = np.hstack((labels_a, labels_b))

atf = np.square(3.9 - a[0,:]) + np.square(6 - a[1,:]) * 0.01
btf = np.square(3.9 - a[0,:]) + np.square(6 - a[1,:]) * 0.01
data_z = np.hstack((atf, btf))

data = np.hstack((a, b))
data = np.vstack((data, data_z))
print("Data", data.shape)
print("A:", a.shape)
print("B:", b.shape)


fig = plt.figure(figsize=(8, 8))
# plt.axis((0, 0, 25, 25))

# ax = fig.add_subplot(111, projection='3d')
# ax.tick_params(labelsize=10)
# ax.set_xlim3d(-40,40)
# ax.set_ylim3d(-40,40)
# ax.set_zlim3d(-40,40)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# ax.plot(a[0, :], a[1, :], atf, 'bo', markersize=2)
# ax.plot(b[0, :], b[1, :], btf, 'ro', markersize=2)

s = 3.9
t = np.linspace(-6 + s,8 + s, 1000)
m = np.square(t - s) * 0.9

plt.plot(t,m, 'black')

plt.plot(a[0, :], a[1, :], 'bo', markersize=2)
plt.plot(b[0, :], b[1, :], 'ro', markersize=2)

plt.title('Separting Trajectories of Moons of Jupiter (Ganymede and Callisto)')
plt.xlabel("Telescope X-axis")
plt.ylabel("Telescope Y-axis")

plt.show()

# Size is the only change when changing #dimensions of the problem |w| = d + 1
w = np.array([5, -7, 5, 5])
lr = 0.001
bias = 0


def update_pla(x, y, w, bias):
    value = w.dot(x) * y + bias
    if value <= 0:
        w = w + (y) * x
        bias = bias + y
    return value, w, bias


def update_svm(x, y, w, ep):
    X = np.hstack((x, -1))
    value = np.dot(X, w)
    d = 1 - (y * value)
    lam = 0.1
    if np.maximum(d, 0) >= 0:
        # print("Above:", np.maximum(d,0))
        g = 2 * lam * w
        err = 2 * lam * np.linalg.norm(w[:-1]) ** 2
    else:
        # print("Below:", np.maximum(d, 0))
        g = 2 * lam * w - (y * X)
        err = y * np.dot(X, w)

    w = w - (lr * g)
    return w, err

def test_simple_models(w):
    cost_history = []
    for i in range(30):

        total_error = 0
        for j in range(n):
            index = np.random.randint(0, n)
            x = data[:, index]
            y = labels[index]

            # value, w, bias = update_pla(x, y, w, bias)
            w, err = update_svm(x, y, w, i + 1)
            total_error += err
            bias = -w[2]

        # plt.plot(a[0, :], a[1, :], 'bo', markersize=2)
        # plt.plot(b[0, :], b[1, :], 'ro', markersize=2)
        # plt.xlim(-8, 12), plt.ylim(-20, 20)

        ax.plot(a[0,:],a[1,:],atf, 'bo', markersize=2)
        ax.plot(b[0,:],b[1,:],btf, 'ro', markersize=2)

        xx, yy = np.meshgrid(np.arange(-10,15,0.5), np.arange(-10,40,0.5))
        z = (-w[0] * xx - w[1] * yy - w[3]) * (1. / w[2])

        ax.plot_surface(xx, yy, z)

        # plt.plot([la[0], lb[0]], [la[1], lb[1]], 'g-', lw=2)
        print(total_error, w)
        cost_history.append(total_error)

        plt.pause(0.01)
        plt.cla()
        # plt.title('Separting Trajectories of Moons of Jupiter (Ganymede and Callisto)')
        # plt.xlabel("Telescope X-axis")
        # plt.ylabel("Telescope Y-axis")

    # plt.clf()
    # plt.plot(cost_history, 'g-')
    # plt.title('Graduate School Acceptance Prediction (Total Cost Per Epoch)')
    # plt.xlabel("Epoch (n)")
    # plt.ylabel("Total Cost")
    # plt.show()

    ax.plot(a[0, :], a[1, :], atf, 'bo', markersize=2)
    ax.plot(b[0, :], b[1, :], btf, 'ro', markersize=2)

    xx, yy = np.meshgrid(np.arange(-10, 15, 0.5), np.arange(-10, 40, 0.5))
    z = (-w[0] * xx - w[1] * yy - w[3]) * (1. / w[2])

    ax.plot_surface(xx, yy, z)

    print("Cost History: ", cost_history)
    plt.figure()
    plt.plot([m for m in range(len(cost_history))], cost_history)
    plt.xlabel("Epochs")
    plt.ylabel("Total Training Error")
    plt.show()

# if __name__ == '__main__':
#     test_simple_models(w)