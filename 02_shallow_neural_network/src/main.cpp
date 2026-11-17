#include "Matrix.h"

#include <vector>
#include <iostream>

struct Point
{
	float x{};
	float y{};
};

class ShallowNetwork
{
private:
	int m_hidden_unit_count {};

	Matrix W_in_hid {};  // Weights between input and hidden layers.
	Matrix W_hid_out {}; // Weights between hidden and output layers.
	Matrix b_hid {};     // Hidden unit biases
	float b_out {0.0f};  // Output bias

public:
	explicit ShallowNetwork(int hidden_unit_count)
		: m_hidden_unit_count { hidden_unit_count }
		, W_in_hid { Matrix::random(hidden_unit_count, 1) * 2.0f - 1.0f }
		, W_hid_out { Matrix::random(1, hidden_unit_count) * 2.0f - 1.0f }
		, b_hid { Matrix::zeros(hidden_unit_count, 1) }
	{}

	const Matrix& weights_in_hid() const { return W_in_hid; }
	const Matrix& weights_hid_out() const { return W_hid_out; }
	const Matrix& biases_hid() const { return b_hid; }
	float bias_out() const { return b_out; }

	float feed_forward(float x) const
	{
		Matrix pre_activations { x * W_in_hid + b_hid };
		Matrix activations { pre_activations.applyFunc([](float x) 
			{
				// ReLU activation function -> max(x, 0)
				return (x > 0.0f) ? x : 0.0f;
			})};

		return Matrix::dot(W_hid_out, activations)(0, 0) + b_out;
	}

	float LSE_loss(const std::vector<Point>& data_set) const
	{
		float total_loss { 0.0f };
		for (const Point& p : data_set)
		{
			float diff {feed_forward(p.x) - p.y};
			total_loss += diff * diff;
		}
		return total_loss;
	}

	void train(const std::vector<Point>& data_set, float learning_rate, int epoch)
	{
		auto ReLU_prime = [](float x)
			{
				return (x > 0.0f) ? 1.0f : 0.0f;
			};

		for (int i { 0 }; i < epoch; ++i)
		{
			Matrix W_in_hid_grad { m_hidden_unit_count, 1 };
			Matrix W_hid_out_grad { 1, m_hidden_unit_count };
			Matrix b_hid_grad { m_hidden_unit_count, 1 };
			float b_out_grad { 0.0f };

			for (const Point& p : data_set)
			{
				Matrix pre_activations{ p.x * W_in_hid + b_hid };
				Matrix activations{ pre_activations.applyFunc([](float x)
					{
						// ReLU activation function -> max(x, 0)
						return (x > 0.0f) ? x : 0.0f;
					}) };

				float out { Matrix::dot(W_hid_out, activations)(0, 0) + b_out };

				float out_grad {2.0f * (out - p.y)};
				
				Matrix common_grad { out_grad * W_hid_out.transpose() * pre_activations.applyFunc(ReLU_prime) };

				b_out_grad += out_grad;
				W_hid_out_grad += activations.transpose() * out_grad;
				b_hid_grad += common_grad;
				W_in_hid_grad += common_grad * p.x;
			}

			W_in_hid -= W_in_hid_grad * learning_rate;
			W_hid_out -= W_hid_out_grad * learning_rate;
			b_hid -= b_hid_grad * learning_rate;
			b_out -= b_out_grad * learning_rate;
		}
	}
};

std::ostream& operator<<(std::ostream& out, const ShallowNetwork& sn)
{
	int parameters{ sn.weights_in_hid().size() + sn.weights_hid_out().size() + sn.biases_hid().size() + 1 };
	return out << "[input=1, hidden=" << sn.weights_in_hid().size() << ", output=1, parameters=" << parameters << ']';
}

int main()
{
	constexpr int hidden_units{ 10 };
	constexpr int epoch{ 5000 };
	constexpr float learning_rate{ 0.005f };

	std::vector<Point> data_set;

	for (float x{ -2.0f }; x <= 2.0f; x += 0.5f)
	{
		data_set.push_back(Point{ x, 0.5f * x * x });
	}

	ShallowNetwork shallowNetwork { hidden_units };

	std::cout << "NETWORK INFO: " << shallowNetwork << "\n\n";
	std::cout << "------------------------- BEFORE TRAINING -------------------------\n\n";
	std::cout << "weights_input_hidden:\t" << shallowNetwork.weights_in_hid().transpose();
	std::cout << "weights_hidden_out:\t" << shallowNetwork.weights_hid_out();
	std::cout << "biases_hidden:\t\t" << shallowNetwork.biases_hid().transpose();
	std::cout << "bias_out:\t\t[ " << shallowNetwork.bias_out() << " ]\n";
	std::cout << "\nLSE Loss:\t" << shallowNetwork.LSE_loss(data_set) << "\n\n";
	
	std::cout << "-------------------------     TRAINING    -------------------------\n\n";

	shallowNetwork.train(data_set, learning_rate, epoch);
	std::cout << "Trained with -> learning_rate: " << learning_rate << ", epoch: " << epoch << "\n\n";
	
	std::cout << "------------------------- AFTER TRAINING  -------------------------\n\n";
	std::cout << "weights_input_hidden:\t" << shallowNetwork.weights_in_hid().transpose();
	std::cout << "weights_hidden_out:\t" << shallowNetwork.weights_hid_out();
	std::cout << "biases_hidden:\t\t" << shallowNetwork.biases_hid().transpose();
	std::cout << "bias_out:\t\t[ " << shallowNetwork.bias_out() << " ]\n";
	std::cout << "\nLSE Loss:\t" << shallowNetwork.LSE_loss(data_set) << "\n\n";

	std::cout << "-------------------------    TESTING     -------------------------\n\n";
	std::cout << "INPUT\t|\tDATA\t|\tGUESS\n";
	std::cout << "\t|\t\t|\t\n";
	for (const Point& p : data_set)
	{
		std::cout << p.x << "\t|\t" << p.y << "\t|\t" << shallowNetwork.feed_forward(p.x) << '\n';
	}

	return 0;
}