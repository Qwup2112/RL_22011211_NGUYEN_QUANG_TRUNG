"""Bai 8 - Discounted return voi nhieu gia tri gamma."""

from bai07 import compute_return


def main() -> None:
    rewards = [1, 1, 1, 1, 1]
    gammas = [0.0, 0.5, 0.9, 0.99, 1.0]

    print("rewards =", rewards)
    print()
    print("| Gamma | Return |")
    print("|------:|-------:|")
    for gamma in gammas:
        print(f"| {gamma:5.2f} | {compute_return(rewards, gamma):6.4f} |")

    print()
    # Voi chuoi reward hang so bang 1, G_0 la tong cua cap so nhan:
    #   gamma = 0    -> chi con reward dau tien = 1
    #   gamma = 0.5  -> 1 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
    #   gamma = 1    -> 5
    print("Nhan xet: gamma cang nho thi agent cang chi quan tam reward truoc mat;")
    print("gamma = 0 bien G_0 thanh dung reward cua buoc ke tiep.")


if __name__ == "__main__":
    main()
